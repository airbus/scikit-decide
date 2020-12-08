int: max_capacity;
int: nb_items;
set of int: ITEMS = 1..nb_items+1;
set of int: ITEMS_VALUE = 0..nb_items;
int: DUMMY_ITEM = 0;
array[ITEMS] of int: values;
array[ITEMS] of int: weights;
array[ITEMS] of var ITEMS_VALUE: list_items;
include "globals.mzn";
include "decreasing.mzn";
include "alldifferent_except_0.mzn";
constraint decreasing(list_items);
constraint alldifferent_except_0(list_items);
var int: value;
constraint sum(i  in ITEMS)(values[list_items[i]+1])==value;
constraint sum(i  in ITEMS)(weights[list_items[i]+1])<=max_capacity;
solve maximize(value);
output["value=\(value)"++
       "items=\(list_items)"];