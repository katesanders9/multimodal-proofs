decomp_prove(Hypothesis, Proof, Score, Depth, MaxNumProofs) :-
	Depth > 1, current_time(CurrentTime), writenl('checking time from decomp 1'), \+timer_finished(CurrentTime),
	writenl('Decomp Proving: ', Hypothesis, 'Depth: ', Depth),
	drg(Hypothesis, 'none', 'ANY', 'ANY',  Candidates),
	((mass_double_unify_check(Candidates, F1, F1Score, F1Proof, F2, F2Score, F2Proof),
	writenl('(outer) mass_double_unify_checked:', F1, F1Score, F1Proof, F2, F2Score, F2Proof));
	(
		\+mass_double_unify_check(Candidates, _, _, _, _, _, _),
		writenl('starting individual decomp fact proofs'),
		member(R, Candidates),
		R = [F1, F2],
		not_proved_n(Hypothesis, MaxNumProofs),
		writenl(Hypothesis, '--F1Rel--', F1Rel, '-->', F1, ',', F2),
		Depth2 is Depth - 1,
		prove(F1, F1Proof, F1Score, Depth2, MaxNumProofs),
		writeln('proved f1: ', F1, F1Proof, F1Score, Depth2),
		((recorded(Hypothesis, [CurrentBestScore, _]), F1Score > CurrentBestScore);
		\+recorded(Hypothesis, _)),
		prove(F2, F2Proof, F2Score, Depth2, MaxNumProofs),
		writeln('proved f2: ', F2, F2Proof, F2Score, Depth2)
	)),
	Score is min(F1Score, F2Score),
	record_proof_if_best(Hypothesis, Score),
	Proof = [[Hypothesis,Score], F1Proof, F2Proof],
	writenl(Hypothesis, ':-', [F1, F2], '(Score: ', Score, ')').

retrieve_prove(Hypothesis, Proof, Score, Depth, MaxNumProofs) :-
	Depth > 1, current_time(CurrentTime), writenl('checking time from retrieve 1'), \+timer_finished(CurrentTime),
	writenl('Retrieve-and-write Proving: ', Hypothesis, 'Depth: ', Depth),
	weak_retrieval(Hypothesis, 'support_fact', .5, Matches),
	member(RetrievedMatch, Matches),
	current_time(CurrentTime2),  \+timer_finished(CurrentTime2),
	not_proved_n(Hypothesis, MaxNumProofs),
	RetrievedMatch = [F1, F1Score],
	writenl('checking time from retrieve 2'), \+timer_finished(CurrentTime),
	drg(Hypothesis, F1, 'ANY', 'ANY', Candidates),
	length(Candidates, L), L > 0,
	py_unzip3(Candidates, F1List, F2List, DRGScores),
	(
		(mass_single_unify_check(F2List, F2, F2Score, F2Proof), member([_, F2, DRGScore], Candidates));
		(
			% writenl('running mass single check failed, decomposing...'),
			\+mass_single_unify_check(F2List, _, _, _),
			member(R, Candidates),
			writenl('R', R),
			R = [_, F2, DRGScore],
			not_proved_n(Hypothesis, MaxNumProofs),
			current_time(CurrentTime3),
			\+timer_finished(CurrentTime3),
			writenl(Hypothesis, '-retriever->', [F1, F2]),
			Depth2 is Depth - 1,
			prove(F2, F2Proof, F2Score, Depth2, 1)
		)
	),
	Score is min(DRGScore, F2Score),
	record_proof_if_best(Hypothesis, Score),
	Proof = [[Hypothesis, Score], [F1], F2Proof],
	writenl(Hypothesis, ':-', [F1, F2], '(Score: ', Score, ')').

%prove(Hypothesis, Proof, Score, Depth, MaxNumProofs) :-
%	\+unify_prove(Hypothesis, _, _, Depth),
%	decomp_prove(Hypothesis, Proof, Score, Depth, MaxNumProofs).

% prove(Hypothesis, Proof, Score, Depth, MaxNumProofs) :-
% 	\+unify_prove(Hypothesis, _, _, Depth),
% 	retrieve_prove(Hypothesis, Proof, Score, Depth, MaxNumProofs).


%combined_prove(Hypothesis, Proof, Score, Depth, MaxNumProofs) :-
%	Depth > 1, current_time(CurrentTime), \+timer_finished(CurrentTime),
%	writenl('Combined Proving: ', Hypothesis, 'Depth: ', Depth),
%	weak_retrieval(Hypothesis, 'support_fact', 75, RetrievedDecompositionList),