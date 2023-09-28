def sample_h(x, n, q):
    out = []
    for i in range(len(x) - n + 1):
        t = []
        for j in range(n):
            t.append(x[i+j])
        out.append((t,(q,'\n'.join(t))))
    return out

def remove_breaks(l):
    while '\n' in l:
        ind = l.index('\n')
        l = l[:ind] + ' ' + l[ind+1:]
    return l

def q2v(q, t):
    names = [x for x in t if x.startswith('(') and ':)' in x]
    names = list(set([x[x.index('(')+1:x.index(':')].lower() for x in names]))
    ql = q.lower()
    q_names = [n for n in names if n in ql]
    return q_names

def get_labels(n):
    all_vals = [i for i in list(n.values()) if i]
    return list(set(all_vals))