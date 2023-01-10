time_limit(X) :- 2 = 1.
start_time(X) :- 2 = 1.
strategies(X) :- 2 = 1.

start_timer(TimeLimit) :-
	py_assertz(time_limit(TimeLimit)), % in seconds
	current_time(StartTime),
	forall(start_time(PrevStartTime), py_retract(start_time(PrevStartTime))),
	py_assertz(start_time(StartTime)).

timer_finished(CurrentTime) :-
	start_time(StartTime), time_limit(TimeLimit),
	ElapsedTime is CurrentTime - StartTime,
	writenl('it has been', ElapsedTime, 'seconds'),
	ElapsedTime > TimeLimit,
	writenl('timer finished with time', ElapsedTime, 'with time limit', TimeLimit).

timeout_prove(Hypothesis, Proof, Score, Depth, MaxNumProofs, TimeLimit) :-
	start_timer(TimeLimit),
	prove(Hypothesis, Proof, Score, Depth, MaxNumProofs, Hypothesis).


bfs_prove(Hypothesis, Proof, Score, Depth, MaxNumProofs) :-
	between(1, Depth, Y),
	writenl('maybe starting proof depth ', Y),
	LessThanY is Y - 1,
	\+recorded(Hypothesis, _),
	writenl('no proofs yet, starting search of depth ', Y),
	prove(Hypothesis, Proof, Score, Y, MaxNumProofs, Hypothesis).

bfs_prove_no_cut(Hypothesis, Proof, Score, Depth, MaxNumProofs) :-
	between(1, Depth, Y),
	writenl('maybe starting proof depth ', Y),
	LessThanY is Y - 1,
	prove(Hypothesis, Proof, Score, Y, MaxNumProofs, Hypothesis).



timeout_bfs_prove(Hypothesis, Proof, Score, Depth, MaxNumProofs, TimeLimit) :-
	start_timer(TimeLimit),
	bfs_prove(Hypothesis, Proof, Score, Depth, MaxNumProofs).

timeout_bfs_prove_no_cut(Hypothesis, Proof, Score, Depth, MaxNumProofs, TimeLimit) :-
	start_timer(TimeLimit),
	bfs_prove_no_cut(Hypothesis, Proof, Score, Depth, MaxNumProofs).