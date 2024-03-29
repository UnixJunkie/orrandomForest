open Printf

module Log = Dolog.Log

let with_in_file fn f =
  let input = open_in_bin fn in
  let res = f input in
  close_in input;
  res

let with_out_file fn f =
  let output = open_out_bin fn in
  let res = f output in
  close_out output;
  res

let append_file_to_buffer buff fn =
  with_in_file fn (fun input ->
      let len = in_channel_length input in
      Buffer.add_channel buff input len
    )

let ignore_fst _fst snd =
  snd

let fold_on_lines_of_file fn f acc =
  with_in_file fn (fun input ->
      let acc' = ref acc in
      try
        while true do
          acc' := f !acc' (input_line input)
        done;
        assert(false)
      with End_of_file -> !acc'
    )

let float_list_of_file fn =
  let res =
    fold_on_lines_of_file fn (fun acc line ->
        let pred =
          try Scanf.sscanf line "%f" (fun x -> x)
          with Scanf.Scan_failure msg ->
            (* percolate a NaN rather than crashing *)
            (Log.error "%s: %s" msg line;
             nan) in
        pred :: acc
      ) [] in
  List.rev res

type filename = string

(* capture everything in case of error *)
let collect_script_and_log
    (debug: bool)
    (r_script_fn: filename) (r_log_fn: filename) (model_fn: filename)
  : Result.t =
  let buff = Buffer.create 4096 in
  bprintf buff "--- %s ---\n" r_script_fn;
  append_file_to_buffer buff r_script_fn;
  bprintf buff "--- %s ---\n" r_log_fn;
  append_file_to_buffer buff r_log_fn;
  let err_msg = Buffer.contents buff in
  if not debug then List.iter Sys.remove [r_script_fn; r_log_fn; model_fn];
  Error err_msg

let read_predictions (debug: bool) (maybe_predictions_fn: Result.t)
  : float list * filename =
  match maybe_predictions_fn with
  | Error err -> failwith err (* should have been handled by user before *)
  | Ok predictions_fn ->
    if debug then Log.debug "%s" predictions_fn;
    (* we did not create that file; we are not responsible for it and will not delete it *)
    (float_list_of_file predictions_fn, predictions_fn)

let run_command ?(debug = false) (cmd: string): unit =
  if debug then Log.info "run_command: %s" cmd;
  match Unix.system cmd with
  | Unix.WSIGNALED _ -> (Log.fatal "run_command: signaled: %s" cmd; exit 1)
  | Unix.WSTOPPED _ -> (Log.fatal "run_command: stopped: %s" cmd; exit 1)
  | Unix.WEXITED i when i <> 0 ->
    (Log.fatal "run_command: exit %d: %s" i cmd; exit 1)
  | Unix.WEXITED _ (* i = 0 then *) -> ()

let array_to_file (fn: filename) (f: 'a -> string) (a: 'a array): unit =
  with_out_file fn (fun out ->
      Array.iter (fun x ->
          fprintf out "%s\n" (f x)
        ) a
    )

let int_of_bool = function
  | true -> 1
  | false -> 0
