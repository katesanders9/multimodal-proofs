''' From Nathaniel's NELLIE codebase. Necessary for problog functionality.'''

from time import time

from problog.extern import problog_export, problog_export_nondet, problog_export_raw
from problog.logic import unquote, make_safe, Term, term2list, list2term, Constant

from problog.logic import Term
from src.utils import flatten

db = problog_export.database


@problog_export_raw("+term")
def py_assertz(term, target=None, **kwargs):
    problog_export.database += term
    target._cache.reset()  # reset tabling cache
    return [(term,)]


@problog_export_raw("+term")
def py_retract(term, target=None, **kwargs):
    db = problog_export.database
    nodekey = db.find(term)
    node = db.get_node(nodekey)
    to_erase = node.children.find(term.args)
    if to_erase:
        item = next(to_erase.__iter__())
        node.children.erase((item,))
        target._cache.reset()  # reset tabling cache
        return [(term,)]
    else:
        return []


@problog_export('+str', '-list')
def drg(hypothesis):
    print(hypothesis, type(hypothesis))
    candidates = [("'fact1'", "'fact2'")]
    ret = [([Term(c_) for c_ in c]) for c in candidates]
    print(ret)
    return ret


@problog_export('+term', "-term")
def debug_fn(term):
    breakpoint()


@problog_export("+term", "-term", "-term")
def py_unzip(l):
    """

    :param l: List Term of 2-Term List Terms
    :return: two List Terms of Terms
    """
    l = [term2list(ll, deep=False)
         for ll in term2list(l, deep=False)]
    ret = tuple([list2term(list(ll)) for ll in zip(*l)])

    return ret


@problog_export("+term", "-term", "-term", "-term")
def py_unzip3(l):
    """
    :param l: List Term of 3-Term List Terms
    :return: two List Terms of Terms
    """
    l = [term2list(ll, deep=False)
         for ll in term2list(l, deep=False)]
    ret = tuple([list2term(list(ll)) for ll in zip(*l)])

    return ret


@problog_export("+term", "+term", "-term")
def py_zip(l, m):
    """
    :param l:  List Terms of Terms
    :param m:  List Terms of Terms
    :return: List Term of 2-Term List Terms
    """
    l = term2list(l, deep=False)
    m = term2list(m, deep=False)
    return list2term([list2term(list(pair)) for pair in zip(l, m)])


@problog_export("+term", "+term", "-term")
def py_append(l, m):
    l = term2list(l, deep=False)
    m = term2list(m, deep=False)
    ret = l + m
    return list2term(ret)


@problog_export("+term", "+term", "-term")
def flatten_zip(l, m):
    """
    Flatten the zipped tuples of two List Terms of List Terms

    :param l:  List Terms of Terms
    :param m:  List Terms of Terms
    :return: List Term of flattened List Terms
    """
    l = [term2list(ll, deep=False) for ll in term2list(l, deep=False)]
    m = [term2list(mm, deep=False) for mm in term2list(m, deep=False)]
    return list2term([list2term(flatten(list(pair))) for pair in zip(l, m)])


@problog_export("+term", "-term", "-term")
def py_halve_list(l):
    l = term2list(l, deep=False)
    half_len = int(len(l) / 2)
    return (list2term(l[:half_len]), list2term(l[half_len:]))


@problog_export("-term")
def current_time():
    return Constant(time())
