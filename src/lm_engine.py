from deepproblog.engines import Engine
from deepproblog.engines.exact_engine import ExactEngine, get_predicate, EXTERN
from problog.extern import problog_export
from problog.logic import Term, Clause, term2list
from problog.program import SimpleProgram

from deepproblog.network import Network


def get_det_predicate(net: Network, engine: Engine):
    def det_predicate(arguments):
        output = net([term2list(arguments, False)])[0]
        return output

    return det_predicate

class LMEngine(ExactEngine):
    def prepare(self, db):
        translated = SimpleProgram()
        for e in db:
            new_es = [e]
            if type(e) is Term or type(e) is Clause:
                p = e.probability
                if p is not None and p.functor == "nn":
                    if len(p.args) == 4:
                        new_es = self.create_nn_predicate_ad(e)
                    elif len(p.args) == 3:
                        new_es = self.create_nn_predicate_det(e)
                    elif len(p.args) == 2:
                        new_es = self.create_nn_predicate_fact(e)
                    else:
                        raise ValueError(
                            "A neural predicate with {} arguments is not supported.".format(
                                len(p.args)
                            )
                        )
            for new_e in new_es:
                translated.add_clause(new_e)
        translated.add_clause(
            Clause(
                Term("_directive"),
                Term("use_module", Term("library", Term("lists.pl"))),
            )
        )
        clause_db = self.engine.prepare(translated)
        problog_export.database = clause_db
        for network in self.model.networks:
            if self.model.networks[network].det:
                signature = ["+term", "-term"]
                func = get_det_predicate(self.model.networks[network], self)
                problog_export(*signature)(
                    func, funcname=EXTERN.format(network), modname=None
                )
            elif self.model.networks[network].k is not None:
                signature = ["+term", "-list"]
                problog_export(*signature)(
                    get_predicate(self.model.networks[network]),
                    funcname="{}_extern_nocache_".format(network),
                    modname=None,
                )

        return clause_db