make_list(0,X,[]).
make_list(Len,X,[X|L]) :-
    Len > 0,
    Len1 is Len-1,
    make_list(Len1,X,L).


dont_prune(Hypothesis, IntermediateScore) :-
	writenl('checking no prune for score ', IntermediateScore),
	(recorded(Hypothesis, [CurrentBestScore, _]), IntermediateScore > CurrentBestScore);
	\+recorded(Hypothesis, _).



% saves "new high score" for hypothesis (or if it has not been saved)
% returns false if not
record_proof_if_best(Hypothesis, Score) :-
	((
	\+recorded(Hypothesis, _, _), CurNumProofs is 0
	); (
		recorded(Hypothesis, CurBest, Ref),
		CurBest = [CurBestScore, CurNumProofs],
		Score >= CurBestScore,
		erase(Ref)
	)
	),
	NewCurNumProofs is CurNumProofs + 1,
	recorda(Hypothesis, [Score, NewCurNumProofs]),
	writenl('New best score for "', Hypothesis, '": ', Score).

not_proved_n(Hypothesis, N) :-
	writenl('checking', Hypothesis,'not proved', N, 'times'),
	\+(recorded(Hypothesis, CurBest, Ref), CurBest = [CurBestScore, CurNumProofs], CurNumProofs > N).