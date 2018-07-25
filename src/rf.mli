type filename = string

(** parameters when training the random forests *)
type params = { ntree: int; (* number of trees *)
                mtry: int; (* number of variables randomly sampled
                              as candidates at each split *)
                importance: bool } (* compute variables' importance *)

(** target usage of the trained model *)
type mode = Regression
          | Classification

(** data layout *)
type nb_columns = int
type sparsity = Dense
              | Sparse of nb_columns

val default_params: int -> mode -> params

val train:
  ?debug:bool ->
  mode ->
  sparsity ->
  params ->
  filename ->
  filename -> Result.t

(** retrieve IncNodePurity for each feature in a trained model.
    Call read_predictions on the result to extract the float values. *)
val get_features_importance:
  ?debug:bool ->
  Result.t ->
  Result.t

val predict:
  ?debug:bool ->
  mode ->
  sparsity ->
  Result.t ->
  filename ->
  Result.t

val read_predictions:
  ?debug:bool ->
  Result.t ->
  float list
