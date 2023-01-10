:- use_module(library(lists)).
:- use_module(library(record)).
:- use_module(library(assert)).
:- use_module('src/problog_fns.py').
:- ['src/problog/utils'].
:- ['src/problog/timer'].
:- ['src/problog/retrieval'].
:- ['src/problog/proof_methods'].

nn(drg_model, [Hypothesis, F1, F1Rel, F2Rel], Candidates) :: drg(Hypothesis, F1, F1Rel, F2Rel, Candidates).
nn(wt_retrieval_model, [HypothesisList, Mode, Threshold], Matches) :: wt_weak_retrieval(HypothesisList, Mode, Threshold, Matches).
% nn(entailment_model, [Hypothesis, Premise], Label) :: entailment(Hypothesis, Premise, Label).

prove(Hypothesis, Proof, Score, Depth, MaxNumProofs, TopHypothesis) :-
	writenl('proving ', Hypothesis),
	\+unify_prove(Hypothesis, _, _, Depth),
	Depth > 0, current_time(CurrentTime), \+timer_finished(CurrentTime),

	% and try generating both (using template guidance)
	drg(Hypothesis, 'none', 'ANY', 'ANY',  GeneratedFactPairsWithScores),

	% first try to retrieve one support fact, generate the other
	retrieve_first_generate_second_fact(Hypothesis, RetrieveGenFactPairsWithScores),
	current_time(CurrentTime5), \+timer_finished(CurrentTime5),
	length(RetrieveGenFactPairsWithScores, L),
	(
		(L > 0, py_unzip3(RetrieveGenFactPairsWithScores, RetrieveGenF1List, RetrieveGenF2List, _));
		(L = 0, RetrieveGenF1List = [], RetrieveGenF2List = [])
	),
	((
	  (
	  	% check if any second facts unify against db
	    mass_single_unify_check(RetrieveGenF2List, F2, F2Score, F2Proof),
	    member(MatchedPair, RetrieveGenFactPairsWithScores),
	    MatchedPair = [F1, F2, DRGScore],
		Score is min(DRGScore, F2Score),
		record_proof_if_best(Hypothesis, Score),
	    Proof = [[Hypothesis, Score], [F1], F2Proof],
	    writenl('Proof1:', Proof)
	  );
	  (
		% and check if generated pairs unify
		length(GeneratedFactPairsWithScores, GFPL), GFPL > 0,
		py_unzip3(GeneratedFactPairsWithScores, GenFact1List, GenFact2List, _),
		py_zip(GenFact1List, GenFact2List, GeneratedFactPairs),
	    mass_double_unify_check(GeneratedFactPairs, F1, F1Score, F1Proof, F2, F2Score, F2Proof),
	    member(UnifiedPair, GeneratedFactPairsWithScores),
	    UnifiedPair = [F1, F2, DRGScore],
		min_list([DRGScore, F1Score, F2Score], Score),
		record_proof_if_best(Hypothesis, Score),
		Proof = [[Hypothesis, Score], F1Proof, F2Proof],
		writenl('Proof2:', Proof)
	  )
	); ( % if no proofs found, recur
	  % recur only if not proved yet
	  % \+recorded(Hypothesis, _),
	  ((max_1_on_recursive_calls(1), RecursiveMaxNumProofs is 1);
	  \+max_1_on_recursive_calls(1), RecursiveMaxNumProofs is MaxNumProofs),
	  ((
		member(R, RetrieveGenFactPairsWithScores),
		R = [F1, F2, DRGScore],
		dont_prune(TopHypothesis, DRGScore),
		dont_prune(Hypothesis, DRGScore),
		not_proved_n(Hypothesis, MaxNumProofs),
		current_time(CurrentTime3), \+timer_finished(CurrentTime3),
		Depth2 is Depth - 1,
		writenl('recurring on f2 of', F2, 'depth', Depth2),
		prove(F2, F2Proof, F2Score, Depth2, RecursiveMaxNumProofs, TopHypothesis),
		writenl('proved f2 (a): ', F2, F2Proof, F2Score, Depth2),
		Score is min(DRGScore, F2Score),
		record_proof_if_best(Hypothesis, Score),
		Proof = [[Hypothesis, Score], [F1], F2Proof],
		writenl('Proof3:', Proof)
	  );
	  (
	    member(R, GeneratedFactPairsWithScores),
		R = [F1, F2, DRGScore],
		dont_prune(TopHypothesis, DRGScore),
		dont_prune(Hypothesis, DRGScore),
		not_proved_n(Hypothesis, MaxNumProofs),
		Depth2 is Depth - 1,
		current_time(CurrentTime4), \+timer_finished(CurrentTime4),
		prove(F1, F1Proof, F1Score, Depth2, RecursiveMaxNumProofs, TopHypothesis),
		writeln('proved f1: ', F1, F1Proof, F1Score, Depth2),
		dont_prune(TopHypothesis, F1Score),
		dont_prune(Hypothesis, F1Score),
		prove(F2, F2Proof, F2Score, Depth2, RecursiveMaxNumProofs, TopHypothesis),
		writeln('proved f2: (b) ', F2, F2Proof, F2Score, Depth2),
		min_list([DRGScore, F1Score, F2Score], Score),
		record_proof_if_best(Hypothesis, Score),
		Proof = [[Hypothesis, Score], F1Proof, F2Proof],
		writenl('Proof4:', Proof)
	  ))
	)).


max_1_on_recursive_calls(1) :- 1 = 2.