using_eqasc(X) :- 1 = 2.

eqasc_weak_retrieval(HypothesisList, Mode, Threshold, Matches) :-
 	\+using_eqasc(1),
	length(HypothesisList, NumHypotheses),
	make_list(NumHypotheses, [], Matches).

weak_retrieval(HypothesisList, Mode, Threshold, Matches) :-
	is_list(HypothesisList), ground(HypothesisList),
	wt_weak_retrieval(HypothesisList, Mode, Threshold, WTMatches),
	eqasc_weak_retrieval(HypothesisList, Mode, Threshold, QASCMatches),
	flatten_zip(WTMatches, QASCMatches, Matches),
	py_zip(HypothesisList, Matches, HFPairList),
	forall(
		member([Hypothesis, Match], HFPairList),
		assert_record(Hypothesis, Mode, weak_retrieval(Hypothesis, Mode, Threshold, Match))
	).

assert_record(H, Mode, Trm) :-
	py_assertz(retrieved(H, Mode)),
	py_assertz(Trm).

retrieved('none', 'none').

weak_retrieval(Hypothesis, Mode, Threshold, Matches) :-
	% ensure that H has not already been checked against KB
	\+is_list(Hypothesis), \+retrieved(Hypothesis, Mode),
	weak_retrieval([Hypothesis], Mode, Threshold, MatchesList), member(Matches, MatchesList).


retrieve_first_generate_second_fact(Hypothesis, OutFactPairsWithScores) :-
	writenl('retrieve-then-gen for ', Hypothesis),
	weak_retrieval(Hypothesis, 'support_fact', .25, RetrievalMatches),
	current_time(CurrentTime),
	writenl('checking time from retrieve then generate'),
	\+timer_finished(CurrentTime),
	length(RetrievalMatches, LRM),
	(
		(LRM > 0,
		py_unzip(RetrievalMatches, F1List, F1Scores),
		findall(
				Candidate,
				(
					% member(RetrievedMatch, RetrievalMatches),
					% RetrievedMatch = [F1, F1Score],
					drg(Hypothesis, F1List, 'ANY', 'ANY', Candidates),
					% writenl('retrieved candidates: ', Candidates),
					% length(Candidates, L), L > 0,
					member(Candidate, Candidates)
				),
				OutFactPairsWithScores
			)
			% , writenl('created the following ret-gen pairs', OutFactPairsWithScores)
			);
		(LRM = 0, OutFactPairsWithScores = [])
	).

mass_double_unify_check(Candidates, Fact1, F1Score, F1Proof, Fact2, F2Score, F2Proof) :-
	% writenl('Mass double unify checking over ', Candidates), % list of fact pairs
	length(Candidates, L), L > 0,
	py_unzip(Candidates, Fact1List, Fact2List),
	py_append(Fact1List, Fact2List, AllFactList),
	weak_retrieval(AllFactList, 'entailment', 0, AllFactMatchList),
	py_halve_list(AllFactMatchList, Fact1MatchList, Fact2MatchList),
	py_zip(Fact1MatchList, Fact2MatchList, CandidateMatchList),
	py_zip(Candidates, CandidateMatchList, CandidatePlusMatchPairList),
	member([[Fact1, Fact2], [F1Matches, F2Matches]], CandidatePlusMatchPairList),
	member(F1Match, F1Matches), member(F2Match, F2Matches),
	F1Match = [F1Provenance, F1Score],
	F2Match = [F2Provenance, F2Score],
	F1Proof = [[Fact1, F1Score], [F1Provenance]],
	F2Proof = [[Fact2, F2Score], [F2Provenance]].


mass_single_unify_check(Candidates, Fact1, F1Score, F1Proof) :-
	% writenl('Mass unify checking over ', Candidates),
	length(Candidates, L), L > 0,
	weak_retrieval(Candidates, 'entailment', 0, MatchesList),
	py_zip(Candidates, MatchesList, CandidatePlusMatchesList),
	member([Fact1, F1Matches], CandidatePlusMatchesList), member(F1Match, F1Matches),
	writenl(Fact1, ':-', F1Match),
	F1Match = [F1Provenance, F1Score],
	F1Proof = [[Fact1, F1Score], [F1Provenance]].