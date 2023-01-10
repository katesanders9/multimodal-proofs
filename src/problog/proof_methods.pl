unify_prove(Hypothesis, Proof, Score, Depth) :-
	writenl('Unification Proving: ', Hypothesis, 'Depth: ', Depth),
	Depth > 0,
	current_time(CurrentTime),  \+timer_finished(CurrentTime),
	weak_retrieval(Hypothesis, 'entailment', 0, Matches),
	writenl(Matches),
	member(Match, Matches),
	not_proved_n(Hypothesis, 1),
	Match = [Provenance, Score],
 	Proof = [[Hypothesis, Score], [Provenance]],
 	record_proof_if_best(Hypothesis, Score).

prove(Hypothesis, Proof, Score, Depth, MaxNumProofs) :-
	unify_prove(Hypothesis, Proof, Score, Depth).

prove(Hypothesis, Proof, Score, Depth, MaxNumProofs, _) :-
	unify_prove(Hypothesis, Proof, Score, Depth).