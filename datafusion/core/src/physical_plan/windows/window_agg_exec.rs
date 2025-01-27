// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

//! Stream and channel implementations for window function expressions.

use crate::dataframe::DataFrame;
use crate::error::Result;
use crate::execution::context::TaskContext;
use crate::physical_plan::expressions::PhysicalSortExpr;
use crate::physical_plan::metrics::{
    BaselineMetrics, ExecutionPlanMetricsSet, MetricsSet,
};
use crate::physical_plan::{
    common, ColumnStatistics, DisplayFormatType, Distribution, ExecutionPlan,
    Partitioning, RecordBatchStream, SendableRecordBatchStream, Statistics, WindowExpr,
};
use crate::prelude::{CsvReadOptions, SessionContext};
use arrow::{
    array::ArrayRef,
    datatypes::{Schema, SchemaRef},
    error::{ArrowError, Result as ArrowResult},
    record_batch::RecordBatch,
};
use futures::stream::Stream;
use futures::{ready, StreamExt};
use std::any::Any;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};

/// Window execution plan
#[derive(Debug)]
pub struct WindowAggExec {
    /// Input plan
    input: Arc<dyn ExecutionPlan>,
    /// Window function expression
    window_expr: Vec<Arc<dyn WindowExpr>>,
    /// Schema after the window is run
    schema: SchemaRef,
    /// Schema before the window
    input_schema: SchemaRef,
    /// Execution metrics
    metrics: ExecutionPlanMetricsSet,
}

impl WindowAggExec {
    /// Create a new execution plan for window aggregates
    pub fn try_new(
        window_expr: Vec<Arc<dyn WindowExpr>>,
        input: Arc<dyn ExecutionPlan>,
        input_schema: SchemaRef,
    ) -> Result<Self> {
        let schema = create_schema(&input_schema, &window_expr)?;
        let schema = Arc::new(schema);
        Ok(Self {
            input,
            window_expr,
            schema,
            input_schema,
            metrics: ExecutionPlanMetricsSet::new(),
        })
    }

    /// Window expressions
    pub fn window_expr(&self) -> &[Arc<dyn WindowExpr>] {
        &self.window_expr
    }

    /// Input plan
    pub fn input(&self) -> &Arc<dyn ExecutionPlan> {
        &self.input
    }

    /// Get the input schema before any window functions are applied
    pub fn input_schema(&self) -> SchemaRef {
        self.input_schema.clone()
    }
}

impl ExecutionPlan for WindowAggExec {
    /// Return a reference to Any that can be used for downcasting
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn schema(&self) -> SchemaRef {
        self.schema.clone()
    }

    fn children(&self) -> Vec<Arc<dyn ExecutionPlan>> {
        vec![self.input.clone()]
    }

    /// Get the output partitioning of this plan
    fn output_partitioning(&self) -> Partitioning {
        // because we can have repartitioning using the partition keys
        // this would be either 1 or more than 1 depending on the presense of
        // repartitioning
        self.input.output_partitioning()
    }

    fn output_ordering(&self) -> Option<&[PhysicalSortExpr]> {
        self.input.output_ordering()
    }

    fn maintains_input_order(&self) -> bool {
        true
    }

    fn relies_on_input_order(&self) -> bool {
        true
    }

    fn required_child_distribution(&self) -> Distribution {
        if self
            .window_expr()
            .iter()
            .all(|expr| expr.partition_by().is_empty())
        {
            Distribution::SinglePartition
        } else {
            Distribution::UnspecifiedDistribution
        }
    }

    fn with_new_children(
        self: Arc<Self>,
        children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        Ok(Arc::new(WindowAggExec::try_new(
            self.window_expr.clone(),
            children[0].clone(),
            self.input_schema.clone(),
        )?))
    }

    fn execute(
        &self,
        partition: usize,
        context: Arc<TaskContext>,
    ) -> Result<SendableRecordBatchStream> {
        let input = self.input.execute(partition, context)?;
        let stream = Box::pin(WindowAggStream::new(
            self.schema.clone(),
            self.window_expr.clone(),
            input,
            BaselineMetrics::new(&self.metrics, partition),
        ));
        Ok(stream)
    }

    fn fmt_as(
        &self,
        t: DisplayFormatType,
        f: &mut std::fmt::Formatter,
    ) -> std::fmt::Result {
        match t {
            DisplayFormatType::Default => {
                write!(f, "WindowAggExec: ")?;
                let g: Vec<String> = self
                    .window_expr
                    .iter()
                    .map(|e| format!("{}: {:?}", e.name().to_owned(), e.field()))
                    .collect();
                write!(f, "wdw=[{}]", g.join(", "))?;
            }
        }
        Ok(())
    }

    fn metrics(&self) -> Option<MetricsSet> {
        Some(self.metrics.clone_inner())
    }

    fn statistics(&self) -> Statistics {
        let input_stat = self.input.statistics();
        let win_cols = self.window_expr.len();
        let input_cols = self.input_schema.fields().len();
        // TODO stats: some windowing function will maintain invariants such as min, max...
        let mut column_statistics = vec![ColumnStatistics::default(); win_cols];
        if let Some(input_col_stats) = input_stat.column_statistics {
            column_statistics.extend(input_col_stats);
        } else {
            column_statistics.extend(vec![ColumnStatistics::default(); input_cols]);
        }
        Statistics {
            is_exact: input_stat.is_exact,
            num_rows: input_stat.num_rows,
            column_statistics: Some(column_statistics),
            // TODO stats: knowing the type of the new columns we can guess the output size
            total_byte_size: None,
        }
    }
}

fn create_schema(
    input_schema: &Schema,
    window_expr: &[Arc<dyn WindowExpr>],
) -> Result<Schema> {
    let mut fields = Vec::with_capacity(input_schema.fields().len() + window_expr.len());
    for expr in window_expr {
        fields.push(expr.field()?);
    }
    fields.extend_from_slice(input_schema.fields());
    Ok(Schema::new(fields))
}

/// Compute the window aggregate columns
fn compute_window_aggregates(
    window_expr: &[Arc<dyn WindowExpr>],
    batch: &RecordBatch,
) -> Result<Vec<ArrayRef>> {
    window_expr
        .iter()
        .map(|window_expr| window_expr.evaluate(batch))
        .collect()
}

/// stream for window aggregation plan
pub struct WindowAggStream {
    schema: SchemaRef,
    input: SendableRecordBatchStream,
    batches: Vec<RecordBatch>,
    finished: bool,
    window_expr: Vec<Arc<dyn WindowExpr>>,
    baseline_metrics: BaselineMetrics,
}

impl WindowAggStream {
    /// Create a new WindowAggStream
    pub fn new(
        schema: SchemaRef,
        window_expr: Vec<Arc<dyn WindowExpr>>,
        input: SendableRecordBatchStream,
        baseline_metrics: BaselineMetrics,
    ) -> Self {
        Self {
            schema,
            input,
            batches: vec![],
            finished: false,
            window_expr,
            baseline_metrics,
        }
    }

    fn compute_aggregates(&self) -> ArrowResult<RecordBatch> {
        // record compute time on drop
        let _timer = self.baseline_metrics.elapsed_compute().timer();

        let batch = common::combine_batches(&self.batches, self.input.schema())?;
        if let Some(batch) = batch {
            // calculate window cols
            let mut columns = compute_window_aggregates(&self.window_expr, &batch)
                .map_err(|e| ArrowError::ExternalError(Box::new(e)))?;

            // combine with the original cols
            // note the setup of window aggregates is that they newly calculated window
            // expressions are always prepended to the columns
            columns.extend_from_slice(batch.columns());
            RecordBatch::try_new(self.schema.clone(), columns)
        } else {
            Ok(RecordBatch::new_empty(self.schema.clone()))
        }
    }
}

impl Stream for WindowAggStream {
    type Item = ArrowResult<RecordBatch>;

    fn poll_next(
        mut self: Pin<&mut Self>,
        cx: &mut Context<'_>,
    ) -> Poll<Option<Self::Item>> {
        let poll = self.poll_next_inner(cx);
        self.baseline_metrics.record_poll(poll)
    }
}

impl WindowAggStream {
    #[inline]
    fn poll_next_inner(
        &mut self,
        cx: &mut Context<'_>,
    ) -> Poll<Option<ArrowResult<RecordBatch>>> {
        if self.finished {
            return Poll::Ready(None);
        }

        loop {
            let result = match ready!(self.input.poll_next_unpin(cx)) {
                Some(Ok(batch)) => {
                    self.batches.push(batch);
                    continue;
                }
                Some(Err(e)) => Err(e),
                None => self.compute_aggregates(),
            };

            self.finished = true;

            return Poll::Ready(Some(result));
        }
    }
}

impl RecordBatchStream for WindowAggStream {
    /// Get the schema
    fn schema(&self) -> SchemaRef {
        self.schema.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::assert_batches_eq;
    use crate::dataframe::DataFrame;
    use crate::datasource::MemTable;
    use crate::prelude::{CsvReadOptions, SessionConfig, SessionContext};
    use arrow::array::{Float32Array, Int64Array};
    use arrow::datatypes::{DataType, Field};
    use arrow::util::pretty;
    use datafusion_common::from_slice::FromSlice;
    use datafusion_common::ScalarValue;

    fn create_ctx() -> SessionContext {
        // define a schema.
        let schema =
            Arc::new(Schema::new(vec![Field::new("a", DataType::Float32, false)]));

        // define data in two partitions
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![Arc::new(Float32Array::from_slice(&[
                1.0, 2.0, 3., 4.0, 5., 6., 7., 8.0,
            ]))],
        )
        .unwrap();

        let batch_2 = RecordBatch::try_new(
            schema.clone(),
            vec![Arc::new(Float32Array::from_slice(&[
                9., 10., 11., 12., 13., 14., 15., 16., 17.,
            ]))],
        )
        .unwrap();
        let ctx = SessionContext::new();
        // declare a new context. In spark API, this corresponds to a new spark SQLsession
        // declare a table in memory. In spark API, this corresponds to createDataFrame(...).
        let provider =
            MemTable::try_new(schema.clone(), vec![vec![batch], vec![batch_2]]).unwrap();
        // Register table
        ctx.register_table("t", Arc::new(provider)).unwrap();
        ctx
    }

    #[tokio::test]
    async fn window_frame_empty() -> Result<()> {
        let ctx = create_ctx();

        // execute the query
        let df = ctx
            .sql("SELECT SUM(a) OVER() as summ, COUNT(*) OVER () as cnt FROM t")
            .await?;

        //let df = df.explain(false, false)?;
        let batches = df.collect().await?;
        pretty::print_batches(&batches).expect("TODO: panic message");
        let expected = vec![
            "+------+-----+",
            "| summ | cnt |",
            "+------+-----+",
            "| 153  | 17  |",
            "| 153  | 17  |",
            "| 153  | 17  |",
            "| 153  | 17  |",
            "| 153  | 17  |",
            "| 153  | 17  |",
            "| 153  | 17  |",
            "| 153  | 17  |",
            "| 153  | 17  |",
            "| 153  | 17  |",
            "| 153  | 17  |",
            "| 153  | 17  |",
            "| 153  | 17  |",
            "| 153  | 17  |",
            "| 153  | 17  |",
            "| 153  | 17  |",
            "| 153  | 17  |",
            "+------+-----+",
        ];
        // The output order is important as SMJ preserves sortedness
        assert_batches_eq!(expected, &batches);
        Ok(())
    }

    #[tokio::test]
    async fn window_frame_rows_preceding() -> Result<()> {
        let ctx = create_ctx();

        // execute the query
        let df = ctx
            .sql(
                "SELECT SUM(a) OVER(ROWS BETWEEN 1 PRECEDING AND 1 FOLLOWING ) as summ FROM t"
            )
            .await?;

        //let df = df.explain(false, false)?;
        let batches = df.collect().await?;
        pretty::print_batches(&batches).expect("TODO: panic message");
        let expected = vec![
            "+------+", "| summ |", "+------+", "| 3    |", "| 6    |", "| 9    |",
            "| 12   |", "| 15   |", "| 18   |", "| 21   |", "| 24   |", "| 27   |",
            "| 30   |", "| 33   |", "| 36   |", "| 39   |", "| 42   |", "| 45   |",
            "| 48   |", "| 33   |", "+------+",
        ];
        // The output order is important as SMJ preserves sortedness
        assert_batches_eq!(expected, &batches);
        Ok(())
    }
    #[tokio::test]
    async fn window_frame_rows_preceding_with_partition() -> Result<()> {
        let schema =
            Arc::new(Schema::new(vec![Field::new("a", DataType::Float32, false)]));

        // define data in two partitions
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![Arc::new(Float32Array::from_slice(&[
                1.0, 2.0, 3., 4.0, 5., 6., 7., 8.0,
            ]))],
        )
        .unwrap();

        let batch_2 = RecordBatch::try_new(
            schema.clone(),
            vec![Arc::new(Float32Array::from_slice(&[
                9., 10., 11., 12., 13., 14., 15., 16., 17.,
            ]))],
        )
        .unwrap();
        let ctx = SessionContext::new();
        // declare a new context. In spark API, this corresponds to a new spark SQLsession
        // declare a table in memory. In spark API, this corresponds to createDataFrame(...).
        let provider = MemTable::try_new(
            schema.clone(),
            vec![
                vec![batch.clone()],
                vec![batch.clone()],
                vec![batch_2.clone()],
                vec![batch_2.clone()],
            ],
        )
        .unwrap();
        // Register table
        ctx.register_table("t", Arc::new(provider)).unwrap();

        // execute the query
        let df = ctx
            .sql(
                "SELECT SUM(a) OVER(PARTITION BY a ROWS BETWEEN 1 PRECEDING AND 1 FOLLOWING ) as summ FROM t"
            )
            .await?;

        //let df = df.explain(false, false)?;
        let batches = df.collect().await?;
        pretty::print_batches(&batches).expect("TODO: panic message");
        let expected = vec![
            "+------+", "| summ |", "+------+", "| 2    |", "| 2    |", "| 4    |",
            "| 4    |", "| 6    |", "| 6    |", "| 8    |", "| 8    |", "| 10   |",
            "| 10   |", "| 12   |", "| 12   |", "| 14   |", "| 14   |", "| 16   |",
            "| 16   |", "| 18   |", "| 18   |", "| 20   |", "| 20   |", "| 22   |",
            "| 22   |", "| 24   |", "| 24   |", "| 26   |", "| 26   |", "| 28   |",
            "| 28   |", "| 30   |", "| 30   |", "| 32   |", "| 32   |", "| 34   |",
            "| 34   |", "+------+",
        ];
        // The output order is important as SMJ preserves sortedness
        assert_batches_eq!(expected, &batches);
        Ok(())
    }

    #[tokio::test]
    async fn window_frame_ranges_preceding_following() -> Result<()> {
        let schema =
            Arc::new(Schema::new(vec![Field::new("a", DataType::Float32, false)]));
        // let schema = Arc::new(Schema::new(vec![Field::new("a", DataType::Int64, false)]));

        // define data in two partitions
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![Arc::new(Float32Array::from_slice(&[1.0, 1.0, 2.0, 3.0]))],
            // vec![Arc::new(Int64Array::from_slice(&[1, 1, 2, 3]))],
        )
        .unwrap();

        let batch_2 = RecordBatch::try_new(
            schema.clone(),
            vec![Arc::new(Float32Array::from_slice(&[5.0, 7.0]))],
            // vec![Arc::new(Int64Array::from_slice(&[5, 7]))],
        )
        .unwrap();
        let ctx = SessionContext::new();
        // declare a new context. In spark API, this corresponds to a new spark SQLsession
        // declare a table in memory. In spark API, this corresponds to createDataFrame(...).

        // let provider = MemTable::try_new(schema.clone(), vec![vec![batch.clone()],vec![batch.clone()], vec![batch_2.clone()], vec![batch_2.clone()]]).unwrap();
        let provider = MemTable::try_new(
            schema.clone(),
            vec![vec![batch.clone()], vec![batch_2.clone()]],
        )
        .unwrap();
        // Register table
        ctx.register_table("t", Arc::new(provider)).unwrap();

        // execute the query
        let df = ctx
            .sql(
                "SELECT SUM(a) OVER(ORDER BY a RANGE BETWEEN 1 PRECEDING AND 1 FOLLOWING ) as summ FROM t"
                // "SELECT SUM(a) OVER(PARTITION BY a ORDER BY a) as summ FROM t"
            )
            .await?;

        //let df = df.explain(false, false)?;
        let batches = df.collect().await?;
        pretty::print_batches(&batches).expect("TODO: panic message");
        let expected = vec![
            "+------+", "| summ |", "+------+", "| 4    |", "| 4    |", "| 7    |",
            "| 5    |", "| 5    |", "| 7    |", "+------+",
        ];
        // The output order is important as SMJ preserves sortedness
        assert_batches_eq!(expected, &batches);
        Ok(())
    }

    #[tokio::test]
    async fn window_frame_empty_inside() -> Result<()> {
        let schema =
            Arc::new(Schema::new(vec![Field::new("a", DataType::Float32, false)]));
        // let schema = Arc::new(Schema::new(vec![Field::new("a", DataType::Int64, false)]));

        // define data in two partitions
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![Arc::new(Float32Array::from_slice(&[1.0, 1.0, 2.0, 3.0]))],
            // vec![Arc::new(Int64Array::from_slice(&[1, 1, 2, 3]))],
        )
        .unwrap();

        let batch_2 = RecordBatch::try_new(
            schema.clone(),
            vec![Arc::new(Float32Array::from_slice(&[5.0, 7.0]))],
            // vec![Arc::new(Int64Array::from_slice(&[5, 7]))],
        )
        .unwrap();
        let ctx = SessionContext::new();
        // declare a new context. In spark API, this corresponds to a new spark SQLsession
        // declare a table in memory. In spark API, this corresponds to createDataFrame(...).

        // let provider = MemTable::try_new(schema.clone(), vec![vec![batch.clone()],vec![batch.clone()], vec![batch_2.clone()], vec![batch_2.clone()]]).unwrap();
        let provider = MemTable::try_new(
            schema.clone(),
            vec![vec![batch.clone()], vec![batch_2.clone()]],
        )
        .unwrap();
        // Register table
        ctx.register_table("t", Arc::new(provider)).unwrap();

        // execute the query
        let df = ctx
            .sql(
                "SELECT SUM(a) OVER() as summ FROM t", // "SELECT SUM(a) OVER(PARTITION BY a ORDER BY a) as summ FROM t"
            )
            .await?;

        //let df = df.explain(false, false)?;
        let batches = df.collect().await?;
        pretty::print_batches(&batches).expect("TODO: panic message");
        let expected = vec![
            "+------+", "| summ |", "+------+", "| 19   |", "| 19   |", "| 19   |",
            "| 19   |", "| 19   |", "| 19   |", "+------+",
        ];
        // The output order is important as SMJ preserves sortedness
        assert_batches_eq!(expected, &batches);
        Ok(())
    }
    #[tokio::test]
    async fn window_frame_rows_preceding_multiple_columns() -> Result<()> {
        let schema = Arc::new(Schema::new(vec![
            Field::new("a", DataType::Float32, false),
            Field::new("b", DataType::Float32, false),
        ]));

        // define data in two partitions
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(Float32Array::from_slice(&[1.0, 2.0, 3., 4., 5.])),
                Arc::new(Float32Array::from_slice(&[2.0, 1., 12., 5., 8.])),
            ],
        )
        .unwrap();

        let ctx = SessionContext::new();
        // declare a new context. In spark API, this corresponds to a new spark SQLsession
        // declare a table in memory. In spark API, this corresponds to createDataFrame(...).
        let provider = MemTable::try_new(schema.clone(), vec![vec![batch]]).unwrap();
        // Register table
        ctx.register_table("t", Arc::new(provider)).unwrap();

        // execute the query
        let df = ctx
            .sql(
                "SELECT corr(a,b) OVER(ROWS BETWEEN 1 PRECEDING AND 1 FOLLOWING ) as cor FROM t"
            )
            .await?;

        //let df = df.explain(false, false)?;
        let batches = df.collect().await?;
        pretty::print_batches(&batches).expect("TODO: panic message");
        let expected = vec![
            "+---------------------+",
            "| cor                 |",
            "+---------------------+",
            "| -1                  |",
            "| 0.8219949365267865  |",
            "| 0.3592106040535498  |",
            "| -0.5694947974514994 |",
            "| 0.999999999999999   |",
            "+---------------------+",
        ];
        // The output order is important as SMJ preserves sortedness
        assert_batches_eq!(expected, &batches);
        Ok(())
    }

    #[tokio::test]
    async fn window_frame_order_by_only() -> Result<()> {
        let schema =
            Arc::new(Schema::new(vec![Field::new("a", DataType::Float32, false)]));
        // let schema = Arc::new(Schema::new(vec![Field::new("a", DataType::Int64, false)]));

        // define data in two partitions
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![Arc::new(Float32Array::from_slice(&[1.0, 1.0, 2.0, 3.0]))],
            // vec![Arc::new(Int64Array::from_slice(&[1, 1, 2, 3]))],
        )
        .unwrap();

        let batch_2 = RecordBatch::try_new(
            schema.clone(),
            vec![Arc::new(Float32Array::from_slice(&[5.0, 7.0]))],
            // vec![Arc::new(Int64Array::from_slice(&[5, 7]))],
        )
        .unwrap();
        let ctx = SessionContext::new();
        // declare a new context. In spark API, this corresponds to a new spark SQLsession
        // declare a table in memory. In spark API, this corresponds to createDataFrame(...).

        // let provider = MemTable::try_new(schema.clone(), vec![vec![batch.clone()],vec![batch.clone()], vec![batch_2.clone()], vec![batch_2.clone()]]).unwrap();
        let provider = MemTable::try_new(
            schema.clone(),
            vec![vec![batch.clone()], vec![batch_2.clone()]],
        )
        .unwrap();
        // Register table
        ctx.register_table("t", Arc::new(provider)).unwrap();

        // execute the query
        let df = ctx
            .sql(
                "SELECT SUM(a) OVER(ORDER BY a) as summ FROM t", // "SELECT SUM(a) OVER(PARTITION BY a ORDER BY a) as summ FROM t"
            )
            .await?;

        //let df = df.explain(false, false)?;
        let batches = df.collect().await?;
        pretty::print_batches(&batches).expect("TODO: panic message");
        let expected = vec![
            "+------+", "| summ |", "+------+", "| 2    |", "| 2    |", "| 4    |",
            "| 7    |", "| 12   |", "| 19   |", "+------+",
        ];
        // The output order is important as SMJ preserves sortedness
        assert_batches_eq!(expected, &batches);
        Ok(())
    }

    #[tokio::test]
    async fn window_frame_ranges_unbounded_preceding_following() -> Result<()> {
        let schema =
            Arc::new(Schema::new(vec![Field::new("a", DataType::Float32, false)]));
        // let schema = Arc::new(Schema::new(vec![Field::new("a", DataType::Int64, false)]));

        // define data in two partitions
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![Arc::new(Float32Array::from_slice(&[1.0, 1.0, 2.0, 3.0]))],
            // vec![Arc::new(Int64Array::from_slice(&[1, 1, 2, 3]))],
        )
        .unwrap();

        let batch_2 = RecordBatch::try_new(
            schema.clone(),
            vec![Arc::new(Float32Array::from_slice(&[5.0, 7.0]))],
            // vec![Arc::new(Int64Array::from_slice(&[5, 7]))],
        )
        .unwrap();
        let ctx = SessionContext::new();
        // declare a new context. In spark API, this corresponds to a new spark SQLsession
        // declare a table in memory. In spark API, this corresponds to createDataFrame(...).

        // let provider = MemTable::try_new(schema.clone(), vec![vec![batch.clone()],vec![batch.clone()], vec![batch_2.clone()], vec![batch_2.clone()]]).unwrap();
        let provider = MemTable::try_new(
            schema.clone(),
            vec![vec![batch.clone()], vec![batch_2.clone()]],
        )
        .unwrap();
        // Register table
        ctx.register_table("t", Arc::new(provider)).unwrap();

        // execute the query
        let df = ctx
            .sql(
                "SELECT SUM(a) OVER(ORDER BY a RANGE BETWEEN UNBOUNDED PRECEDING AND 1 FOLLOWING ) as summ FROM t"
                // "SELECT SUM(a) OVER(PARTITION BY a ORDER BY a) as summ FROM t"
            )
            .await?;

        //let df = df.explain(false, false)?;
        let batches = df.collect().await?;
        pretty::print_batches(&batches).expect("TODO: panic message");
        let expected = vec![
            "+------+", "| summ |", "+------+", "| 4    |", "| 4    |", "| 7    |",
            "| 7    |", "| 12   |", "| 19   |", "+------+",
        ];
        // The output order is important as SMJ preserves sortedness
        assert_batches_eq!(expected, &batches);
        Ok(())
    }

    #[tokio::test]
    async fn window_frame_ranges_unbounded_preceding_following_diff_col() -> Result<()> {
        let schema = Arc::new(Schema::new(vec![
            Field::new("a", DataType::Float32, false),
            Field::new("b", DataType::Float32, false),
        ]));
        // let schema = Arc::new(Schema::new(vec![Field::new("a", DataType::Int64, false)]));

        // define data in two partitions
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(Float32Array::from_slice(&[1.0, 1.0, 2.0, 3.0])),
                Arc::new(Float32Array::from_slice(&[7.0, 5.0, 3.0, 2.0])),
            ],
            // vec![Arc::new(Int64Array::from_slice(&[1, 1, 2, 3]))],
        )
        .unwrap();

        let batch_2 = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(Float32Array::from_slice(&[5.0, 7.0])),
                Arc::new(Float32Array::from_slice(&[1.0, 1.0])),
            ],
            // vec![Arc::new(Int64Array::from_slice(&[5, 7]))],
        )
        .unwrap();
        let ctx = SessionContext::new();
        // declare a new context. In spark API, this corresponds to a new spark SQLsession
        // declare a table in memory. In spark API, this corresponds to createDataFrame(...).

        // let provider = MemTable::try_new(schema.clone(), vec![vec![batch.clone()],vec![batch.clone()], vec![batch_2.clone()], vec![batch_2.clone()]]).unwrap();
        let provider = MemTable::try_new(
            schema.clone(),
            vec![vec![batch.clone()], vec![batch_2.clone()]],
        )
        .unwrap();
        // Register table
        ctx.register_table("t", Arc::new(provider)).unwrap();

        // execute the query
        let df = ctx
            .sql(
                "SELECT SUM(a) OVER(ORDER BY b RANGE BETWEEN CURRENT ROW AND 1 FOLLOWING ) as summ FROM t"
                // "SELECT SUM(a) OVER(PARTITION BY a ORDER BY a) as summ FROM t"
            )
            .await?;

        //let df = df.explain(false, false)?;
        let batches = df.collect().await?;
        pretty::print_batches(&batches).expect("TODO: panic message");
        let expected = vec![
            "+------+", "| summ |", "+------+", "| 15   |", "| 15   |", "| 5    |",
            "| 2    |", "| 1    |", "| 1    |", "+------+",
        ];
        // The output order is important as SMJ preserves sortedness
        assert_batches_eq!(expected, &batches);
        Ok(())
    }
}
