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

//! Physical exec for aggregate window function expressions.

use crate::window::partition_evaluator::find_ranges_in_range;
use crate::{expressions::PhysicalSortExpr, PhysicalExpr};
use crate::{window::WindowExpr, AggregateExpr};
use arrow::compute::{cast, concat};
use arrow::record_batch::RecordBatch;
use arrow::{array::ArrayRef, datatypes::Field};
use datafusion_common::{DataFusionError, ScalarValue};
use datafusion_common::Result;
use datafusion_expr::{Accumulator, WindowFrameBound};
use datafusion_expr::{WindowFrame, WindowFrameUnits};
use std::any::Any;
use std::cmp::{max, min};
use std::fmt::Display;
use std::iter::IntoIterator;
use std::ops::Range;
use std::sync::Arc;
use arrow::array::Array;


pub fn bisect_left_arrow(a: &Arc<dyn arrow::array::Array>, len: usize, target_value: f64) -> Option<usize> {
    let mut low: usize = 0;
    let mut high: usize = len as usize;

    while low < high {
        let mid = ((high - low) / 2) + low;
        let mid_index = mid as usize;
        let val = a.as_any().downcast_ref::<arrow::array::Float64Array>().unwrap().value(mid);
        // let val = a.value(mid_index);
        // let val = &a[mid_index];


        // Search values that are greater than val - to right of current mid_index
        if val < target_value {
            low = mid + 1;
        } else{
            high = mid;
        }
    }
    Some(low)
}

pub fn bisect_right_arrow(a: &Arc<dyn arrow::array::Array>, len: usize, target_value: f64) -> Option<usize> {
    let mut low: usize = 0;
    let mut high: usize = len as usize;

    while low < high {
        let mid = ((high - low) / 2) + low;
        let mid_index = mid as usize;
        let val = a.as_any().downcast_ref::<arrow::array::Float64Array>().unwrap().value(mid);
        // let val = &a[mid_index];


        // Search values that are greater than val - to right of current mid_index
        if val > target_value {
            high = mid;
        } else{
            low = mid + 1;
        }
    }
    Some(low)
}

pub fn bisect_left<T: std::cmp::PartialOrd>(a: &[T], len: usize, target_value: &T) -> Option<usize> {
    let mut low: usize = 0;
    let mut high: usize = len as usize;

    while low < high {
        let mid = ((high - low) / 2) + low;
        let mid_index = mid as usize;
        let val = &a[mid_index];


        // Search values that are greater than val - to right of current mid_index
        if val < target_value {
            low = mid + 1;
        } else{
            high = mid;
        }
    }
    Some(low)
}

pub fn bisect_right<T: std::cmp::PartialOrd>(a: &[T], len: usize, target_value: &T) -> Option<usize> {
    let mut low: usize = 0;
    let mut high: usize = len as usize;

    while low < high {
        let mid = ((high - low) / 2) + low;
        let mid_index = mid as usize;
        let val = &a[mid_index];


        // Search values that are greater than val - to right of current mid_index
        if val > target_value {
            high = mid;
        } else{
            low = mid + 1;
        }
    }
    Some(low)
}

pub fn combine_ranges(value_ranges: &[Range<usize>]) -> Range<usize> {
    // make ranges single
    let mut glob_range = Range { start: 0, end: 0 };
    for value_range in value_ranges {
        if value_range.start < glob_range.start {
            glob_range.start = value_range.start;
        }
        if value_range.end > glob_range.end {
            glob_range.end = value_range.end;
        }
    }
    glob_range
}

/// A window expr that takes the form of an aggregate function
#[derive(Debug)]
pub struct AggregateWindowExpr {
    aggregate: Arc<dyn AggregateExpr>,
    partition_by: Vec<Arc<dyn PhysicalExpr>>,
    order_by: Vec<PhysicalSortExpr>,
    window_frame: Option<WindowFrame>,
}

impl AggregateWindowExpr {
    /// create a new aggregate window function expression
    pub fn new(
        aggregate: Arc<dyn AggregateExpr>,
        partition_by: &[Arc<dyn PhysicalExpr>],
        order_by: &[PhysicalSortExpr],
        window_frame: Option<WindowFrame>,
    ) -> Self {
        Self {
            aggregate,
            partition_by: partition_by.to_vec(),
            order_by: order_by.to_vec(),
            window_frame,
        }
    }

    /// the aggregate window function operates based on window frame, and by default the mode is
    /// "range".
    fn evaluation_mode(&self) -> WindowFrameUnits {
        self.window_frame.unwrap_or_default().units
    }

    /// create a new accumulator based on the underlying aggregation function
    fn create_accumulator(&self) -> Result<AggregateWindowAccumulator> {
        let accumulator = self.aggregate.create_accumulator()?;
        let window_frame= self.window_frame;
        let order_by = self.order_by().to_vec();
        Ok(AggregateWindowAccumulator { accumulator, window_frame, order_by })
    }

    /// peer based evaluation based on the fact that batch is pre-sorted given the sort columns
    /// and then per partition point we'll evaluate the peer group (e.g. SUM or MAX gives the same
    /// results for peers) and concatenate the results.
    fn peer_based_evaluate(&self, batch: &RecordBatch) -> Result<ArrayRef> {
        let num_rows = batch.num_rows();
        let partition_points =
            self.evaluate_partition_points(num_rows, &self.partition_columns(batch)?)?;
        // let sort_partition_points = self.evaluate_partition_points(num_rows, &self.sort_columns(batch)?)?;
        let values = self.evaluate_args(batch)?;

        let columns = self.sort_columns(batch)?;
        let array_refs: Vec<&ArrayRef> = columns.iter().map(| s | &s.values).collect();
        // Sort values, this will make the same partitions consecutive.
        let sort_partition_points =
            self.evaluate_partition_points(num_rows, &columns)?;

        let results = partition_points
            .iter()
            .map(|partition_range| {
                let sort_partition_points =
                    find_ranges_in_range(partition_range, &sort_partition_points);
                let mut window_accumulators = self.create_accumulator()?;
                let res = window_accumulators.scan_peers(&values, &array_refs, &sort_partition_points);
                // res.unwrap()
                Ok(vec![res.unwrap()])

                // sort_partition_points
                //     .iter()
                //     .map(|range| window_accumulators.scan_peers(&values, range))
                //     .collect::<Result<Vec<_>>>()
            })
            .collect::<Result<Vec<Vec<ArrayRef>>>>()?
            .into_iter()
            .flatten()
            .collect::<Vec<ArrayRef>>();
        let results = results.iter().map(|i| i.as_ref()).collect::<Vec<_>>();
        concat(&results).map_err(DataFusionError::ArrowError)
    }

    fn group_based_evaluate(&self, _batch: &RecordBatch) -> Result<ArrayRef> {
        Err(DataFusionError::NotImplemented(format!(
            "Group based evaluation for {} is not yet implemented",
            self.name()
        )))
    }

    fn row_based_evaluate(&self, batch: &RecordBatch) -> Result<ArrayRef> {
        let num_rows = batch.num_rows();
        let partition_points =
            self.evaluate_partition_points(num_rows, &self.partition_columns(batch)?)?;
        // Sort values, this will make the same partitions consecutive.
        let sort_partition_points =
            self.evaluate_partition_points(num_rows, &self.sort_columns(batch)?)?;
        // Get necessary column.
        let values = self.evaluate_args(batch)?;

        let columns = self.sort_columns(batch)?;
        let array_refs: Vec<&ArrayRef> = columns.iter().map(| s | &s.values).collect();
        // Sort values, this will make the same partitions consecutive.
        let sort_partition_points =
            self.evaluate_partition_points(num_rows, &columns)?;

        let results = partition_points
            .iter()
            .map(|partition_range| {
                let sort_partition_points =
                    find_ranges_in_range(partition_range, &sort_partition_points);

                let mut window_accumulators = self.create_accumulator()?;

                let res = window_accumulators.scan_peers(&values,&array_refs, &sort_partition_points);
                // res.unwrap()
                Ok(vec![res.unwrap()])

                // sort_partition_points
                //     .iter()
                //     .map(|range| {window_accumulators.scan_peers(&values, range)})
                //     .collect::<Result<Vec<_>>>()
            })
            .collect::<Result<Vec<Vec<ArrayRef>>>>()?
            .into_iter()
            .flatten()
            .collect::<Vec<ArrayRef>>();
        let results = results.iter().map(|i| i.as_ref()).collect::<Vec<_>>();
        concat(&results).map_err(DataFusionError::ArrowError)
    }
}

impl WindowExpr for AggregateWindowExpr {
    /// Return a reference to Any that can be used for downcasting
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn field(&self) -> Result<Field> {
        self.aggregate.field()
    }

    fn name(&self) -> &str {
        self.aggregate.name()
    }

    fn expressions(&self) -> Vec<Arc<dyn PhysicalExpr>> {
        self.aggregate.expressions()
    }

    /// evaluate the window function values against the batch
    fn evaluate(&self, batch: &RecordBatch) -> Result<ArrayRef> {
        match self.evaluation_mode() {
            WindowFrameUnits::Range => self.peer_based_evaluate(batch),
            WindowFrameUnits::Rows => self.row_based_evaluate(batch),
            WindowFrameUnits::Groups => self.group_based_evaluate(batch),
        }
    }

    fn partition_by(&self) -> &[Arc<dyn PhysicalExpr>] {
        &self.partition_by
    }

    fn order_by(&self) -> &[PhysicalSortExpr] {
        &self.order_by
    }
}



fn row_frame_to_scalar_bounds(window_frame: &Option<WindowFrame>) -> (usize, usize){
    let res = window_frame.unwrap();
    let preceding: usize = match window_frame.unwrap().start_bound {
        WindowFrameBound::Preceding(None) => usize::try_from(0).unwrap(),
        WindowFrameBound::Preceding(Some(n)) => usize::try_from(n).unwrap(),
        _ => panic!("sa")
    };

    let following: usize = match window_frame.unwrap().end_bound {
        WindowFrameBound::CurrentRow => usize::try_from(0).unwrap(),
        WindowFrameBound::Following(None) => usize::try_from(0).unwrap(),
        WindowFrameBound::Following(Some(n)) => usize::try_from(n).unwrap(),
        _ => panic!("sa")
    };
    (preceding, following)
}

fn range_frame_to_scalar_bounds(window: WindowFrame) -> (f64, f64) {
    let preceding = match window.start_bound {
        WindowFrameBound::Preceding(None) => 0 as f64,
        WindowFrameBound::Preceding(Some(n)) => n as f64,
        _ => panic!("sa")
    };
    let following = match window.end_bound {
        WindowFrameBound::Following(None) => 0 as f64,
        WindowFrameBound::Following(Some(n)) => n as f64,
        _ => panic!("sa")
    };
    (preceding, following)
}

fn calc_cur_range(a: &Arc<dyn arrow::array::Array>, window: WindowFrame, len:usize, idx:usize) -> (usize, usize){
    let start = match window.start_bound{
        // UNBOUNDED PRECEDING
        WindowFrameBound::Preceding(None) => {
            0
        }
        WindowFrameBound::Preceding(Some(n)) => {
            let val = a.as_any().downcast_ref::<arrow::array::Float64Array>().unwrap().value(idx);
            // let val = val as usize;
            let start_range = val - (n as f64);
            // let end_range = val + following;

            let start = bisect_left_arrow(&a, len, start_range).unwrap();
            start
            // let mut end = bisect_right_arrow(&vec, len, end_range).unwrap();
        }
        WindowFrameBound::CurrentRow=>{
            let n = 0;
            let val = a.as_any().downcast_ref::<arrow::array::Float64Array>().unwrap().value(idx);
            // let val = val as usize;
            let start_range = val - (n as f64);
            // let end_range = val + following;

            let start = bisect_left_arrow(&a, len, start_range).unwrap();
            start
        }
        _ => panic!("Invalid Frame bound")
    };
    let end = match window.end_bound{
        // UNBOUNDED PRECEDING
        WindowFrameBound::Following(None) => {
            len
        }
        WindowFrameBound::Following(Some(n)) => {
            let val = a.as_any().downcast_ref::<arrow::array::Float64Array>().unwrap().value(idx);
            // let val = val as usize;
            // let start_range = val - n;
            let end_range = val + (n as f64);

            // let start = bisect_left_arrow(&vec, len, start_range).unwrap();
            let end = bisect_right_arrow(&a, len, end_range).unwrap();
            end
        }
        WindowFrameBound::CurrentRow=>{
            let n = 0;
            let val = a.as_any().downcast_ref::<arrow::array::Float64Array>().unwrap().value(idx);
            // let val = val as usize;
            // let start_range = val - n;
            let end_range = val + (n as f64);

            // let start = bisect_left_arrow(&vec, len, start_range).unwrap();
            let end = bisect_right_arrow(&a, len, end_range).unwrap();
            end
        }
        _ => panic!("Invalid Frame bound")
    };
    (start, end)
}

/// Aggregate window accumulator utilizes the accumulator from aggregation and do a accumulative sum
/// across evaluation arguments based on peer equivalences.
#[derive(Debug)]
struct AggregateWindowAccumulator {
    accumulator: Box<dyn Accumulator>,
    window_frame: Option<WindowFrame>,
    order_by: Vec<PhysicalSortExpr>
}

impl AggregateWindowAccumulator {
    /// scan one peer group of values (as arguments to window function) given by the value_range
    /// and return evaluation result that are of the same number of rows.
    ///
    fn scan_peers(&mut self,
                  values: &[ArrayRef],
                  order_bys: &Vec<&ArrayRef>,
                  value_ranges: &[Range<usize>]) -> Result<ArrayRef> {
        match self.window_frame{
            None => self.scan_peers_range(values,order_bys, value_ranges),
            Some(value) => match value.units {
                WindowFrameUnits::Range => self.scan_peers_range(values, order_bys, value_ranges),
                WindowFrameUnits::Rows => self.scan_peers_row(values, value_ranges),
                WindowFrameUnits::Groups => self.scan_peers_group(values, value_ranges),
            }
        }
    }

    fn scan_peers_group(
        &mut self,
        _values: &[ArrayRef],
        _value_range: &[Range<usize>],
    ) -> Result<ArrayRef> {
        Err(DataFusionError::NotImplemented(format!(
            "Group based evaluation for is not yet implemented",
        )))
    }
    fn scan_peers_range(
        &mut self,
        values: &[ArrayRef],
        order_bys: &Vec<&ArrayRef>,
        value_ranges: &[Range<usize>],
    ) -> Result<ArrayRef> {

        let value_range = combine_ranges(value_ranges);

        if value_range.is_empty() {
            return Err(DataFusionError::Internal(
                "Value range cannot be empty".to_owned(),
            ));
        }
        let len = value_range.end - value_range.start;
        let values = values
            .iter()
            .map(|v| {
                v.slice(value_range.start, len)
            })
            .collect::<Vec<_>>();

        match order_bys.len() {
            0 => {
                // // OVER() case and OVER(ORDER BY <field>) case
                self.accumulator.update_batch(&values)?;
                let value = self.accumulator.evaluate()?;
                Ok(value.to_array_of_size(len))
            }
            _ => {
                if self.window_frame.is_none() {
                    // OVER(ORDER BY <field>)  case
                    let res = WindowFrame {
                        units: WindowFrameUnits::Range,
                        start_bound: WindowFrameBound::Preceding(None),
                        end_bound: WindowFrameBound::Following(Some(0)),
                    };

                    self.window_frame = Some(res);

                }

                match self.window_frame {
                    None => {
                        // // OVER() case and OVER(ORDER BY <field>) case
                        self.accumulator.update_batch(&values)?;
                        let value = self.accumulator.evaluate()?;
                        Ok(value.to_array_of_size(len))
                    }
                    Some(window) => {
                        let mut scalar_iter = vec![];
                        let mut last_range: (usize, usize) = (0, 0);
                        let vec = &cast(&values[0], &arrow::datatypes::DataType::Float64)?;
                        let use_during_range = &cast(order_bys[0], &arrow::datatypes::DataType::Float64)?;
                        for i in 0..vec.len() {
                            let cur_range = calc_cur_range(&use_during_range, window, len, i);

                            let update: Vec<ArrayRef> = values.iter().map(|v| {
                                v.slice(last_range.1, cur_range.1 - last_range.1)
                            }).collect();
                            let retract: Vec<ArrayRef> = values.iter().map(|v| {
                                v.slice(last_range.0, cur_range.0 - last_range.0)
                            }).collect();

                            self.accumulator.update_batch(&update).expect("TODO: panic message");
                            self.accumulator.retract_batch(&retract).expect("TODO: panic message");
                            last_range = cur_range;
                            println!("state: {}", self.state);
                            scalar_iter.push(self.accumulator.evaluate()?);
                        }

                        let array = ScalarValue::iter_to_array(scalar_iter.into_iter());
                        // res.push(array)
                        array
                    }
                }
            }
        }



    }

    fn scan_peers_row(
        &mut self,
        values: &[ArrayRef],
        value_ranges: &[Range<usize>],
    ) -> Result<ArrayRef> {
        let mut vec = vec![];
        for value_range in value_ranges {
            if value_range.is_empty() {
                return Err(DataFusionError::Internal(
                    "Value range cannot be empty".to_owned(),
                ));
            }
            let len = value_range.end - value_range.start;
            let values = values
                .iter()
                .map(|v| {
                    v.slice(value_range.start, len)
                })
                .collect::<Vec<_>>();

            let (preceding, following): (usize, usize) = row_frame_to_scalar_bounds(&self.window_frame);
            let mut scalar_iter = vec![];
            let mut last_range: (usize, usize) = (0, 0);
            for i in 0..values[0].len() {
                let mut start = match i >= preceding {
                    true => i - preceding,
                    false => 0
                };
                let mut end = min(i + following + 1, values[0].len());
                let mut cur_range = (start, end);

                let update: Vec<ArrayRef> = values.iter().map(|v| {
                    v.slice(last_range.1, cur_range.1 - last_range.1)
                }).collect();
                let retract: Vec<ArrayRef> = values.iter().map(|v| {
                    v.slice(last_range.0, cur_range.0 - last_range.0)
                }).collect();

                self.accumulator.update_batch(&update).expect("TODO: panic message");
                self.accumulator.retract_batch(&retract).expect("TODO: panic message");
                last_range = cur_range;
                scalar_iter.push(self.accumulator.evaluate()?);
            }

            let array = ScalarValue::iter_to_array(scalar_iter.into_iter());
            vec.push(array);
            // array

        }
        vec.pop().unwrap()
        // Err(DataFusionError::Internal(
        //     "Value range cannot be empty".to_owned(),
        // ))
    }
}
