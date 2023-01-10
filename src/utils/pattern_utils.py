''' From Nathaniel's NELLIE codebase.'''

import re

class InferencePattern:
    def __init__(self, pattern):
        self.patterns = pattern
        self.regexes = {k: re.compile(re.sub(r'(<[^>]+>)', "(?P\\1.+)", v)) for (k,v) in self.patterns.items()}

    def match(self, cand_dict):
        def _pull_matches(pat, text):
            result = pat.match(text)
            if result:
                return result.groupdict()
            else:
                return None

        matches = {}

        for k,reg in self.regexes.items():
            if k not in cand_dict:
                raise Exception(f"match key {k} not in candidate dict {cand_dict}")
            matchdict = _pull_matches(reg, cand_dict[k])
            # print(matchdict)
            if matchdict is None:
                return False

            for (mkey, mtext) in matchdict.items():
                if mkey in matches and mtext != matches[mkey]:
                    return False
                else:
                    matches[mkey] = mtext


        return True

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return str(self.patterns)
