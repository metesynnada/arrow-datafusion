use arrow::array::{Array, ArrayData, ArrayRef, Float32Array, Float64Array};
use arrow::compute::cast;
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use datafusion_common::from_slice::FromSlice;
use itertools::{multizip, Itertools, Tuples};
use std::sync::Arc;

fn get_index_slice(input: Vec<&[f64]>, idx: usize) -> Vec<&f64> {
    input.iter().map(|arr| arr.get(idx).unwrap()).collect_vec()
}

pub fn bisect_left_arrow(items: Vec<ArrayRef>, k: Vec<ArrayRef>) -> Option<usize> {
    let mut low: usize = 0;
    let mut high: usize = items.len() - 1;
    let item_arrs = items
        .iter()
        .map(|item| {
            item.as_any()
                .downcast_ref::<Float64Array>()
                .unwrap()
                .values()
        })
        .collect_vec();
    let target_value = k
        .iter()
        .map(|item| {
            item.as_any()
                .downcast_ref::<Float64Array>()
                .unwrap()
                .values()
        })
        .collect_vec()
        .iter()
        .map(|arr| arr.get(0).unwrap())
        .collect_vec();

    while low < high {
        let mid = ((high - low) / 2) + low;
        let val: Vec<&f64> = item_arrs
            .iter()
            .map(|arr| arr.get(mid).unwrap())
            .collect_vec();
        // Search values that are greater than val - to right of current mid_index
        if val < target_value {
            low = mid + 1;
        } else {
            high = mid;
        }
    }
    Some(low)
}

fn main() {
    let arrays: Vec<ArrayRef> = vec![
        Arc::new(Float32Array::from_slice(&[5.0, 7.0, 9., 10.])),
        Arc::new(Float32Array::from_slice(&[2.0, 3.0, 4.0])),
        Arc::new(Float32Array::from_slice(&[5.0, 7.0, 10.])),
        Arc::new(Float32Array::from_slice(&[5.0, 7.0, 10.])),
    ];
    let order_columns: Vec<ArrayRef> = arrays
        .iter()
        .map(|array| cast(&array, &DataType::Float64).unwrap())
        .collect();
    let search_tuple: Vec<ArrayRef> = vec![
        Arc::new(Float32Array::from_slice(&[8.0])),
        Arc::new(Float32Array::from_slice(&[3.0])),
        Arc::new(Float32Array::from_slice(&[8.0])),
        Arc::new(Float32Array::from_slice(&[8.0])),
    ];
    let k: Vec<ArrayRef> = search_tuple
        .iter()
        .map(|array| cast(&array, &DataType::Float64).unwrap())
        .collect();

    let res: usize = bisect_left_arrow(order_columns, k).unwrap();
    // define data in two partitions
    assert_eq!(res, 2);
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::{ArrayRef, Float32Array};
    use arrow::compute::cast;
    use arrow::datatypes::{DataType, Field, Schema};
    use arrow::record_batch::RecordBatch;
    use datafusion_common::ScalarValue;
    use datafusion_common::ScalarValue::Null;
    use std::sync::Arc;

    #[tokio::test]
    async fn test_me() {
        main();
    }

    #[tokio::test]
    async fn vector_ord() {
        assert_eq!(true, vec![1, 1] < vec![1, 2]);
        assert_eq!(
            true,
            vec![1, 0, 0, 0, 0, 0, 0, 1] < vec![1, 0, 0, 0, 0, 0, 0, 2]
        );
        assert_eq!(
            true,
            vec![1, 0, 0, 0, 0, 0, 1, 1] > vec![1, 0, 0, 0, 0, 0, 0, 2]
        );
        assert_eq!(
            true,
            vec![1, 0, 0, 0, 0, 1, 9, 9] < vec![1, 0, 0, 0, 0, 2, 0, 0]
        );
        assert_eq!(
            true,
            vec![
                ScalarValue::Int32(Some(2)),
                Null,
                ScalarValue::Int32(Some(0))
            ] < vec![
                ScalarValue::Int32(Some(2)),
                Null,
                ScalarValue::Int32(Some(1))
            ]
        );
    }
}
