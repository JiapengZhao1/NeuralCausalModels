import itertools

import torch as T

from .distribution import FactorizedDistribution
from .scm import SCM

RPA = {
    'bidirected': dict(X=[], Y=[('X',)]),
    'backdoor': dict(Z=[], X=['Z'], Y=['X', 'Z']),
    'bow': dict(X=[], Y=['X', ('X',)]),
    'frontdoor': dict(X=[], M=['X'], Y=['M', ('X',)]),
    'iv': dict(I=[], X=['I'], Y=['X', ('X',)]),
    'm': dict(X=[], Y=['X'], M=[('X',), ('Y',)]),
    'simple': dict(X=[], Y=['X']),
    'bad_m': dict(X=[], Z=[('X',), ('Y',)], Y=['X', 'Z']),
    'bad_m_2': dict(Z=[('X',), ('Y',)], X=['Z'], Y=['X']),
    'bdm': dict(Z=[('X',), ('Y',)], X=['Z'], Y=['X', 'Z']),
    'extended_bow': dict(X=[], Z=['X', ('X',)], Y=['Z']),
    'chain': dict(X=[], Z=['X'], Y=['Z']),
    'double_bow': dict(X=[], Z=['X', ('X',), ('Y',)], Y=['Z']),
    'napkin': dict(W=[('X',), ('Y',)], Z=['W'], X=['Z'], Y=['X']),
    # ---- exp1-8 reordered for correct sampling ----
    'exp1': dict(
        W=[('X',), ('Y',)],
        R=['W'],
        X=['R'],
        Y=['X']
    ),
    'exp2': dict(
        X=[],
        R=['X'],
        Z=['X', ('X',)],
        X2=['X', 'Z'],
        Y=['X2', 'R', ('X',), ('Z',)]
    ),
    'exp3': dict(
        W=[],
        Y=['W', ('W',)],
        X=['W', ('W',), ('Y',)],
        Z=['W', 'Y', 'X']
    ),
    'exp4': dict(
        X=[],
        R=['X'],
        Z=['X', ('X',)],
        X2=['Z', 'X'],
        X3=['Z', 'X2', ('Z',)],
        S=['X3'],
        Y=['R', 'X2', 'S', ('X',), ('X3',)]
    ),
    'exp5': dict(
        Z2=[],
        X=['Z2', ('Z2',)],
        Z1=['Z2', 'X'],
        Z3=['Z2', ('X',)],
        Y=['Z1', 'Z3', ('Z2',), ('X',)]
    ),
    'exp6': dict(
        Z3=[],
        X2=['Z3'],
        X=['X2', ('X2',)],
        Z1=['X'],
        Z2=['Z3', ('X2',)],
        Y=['Z1', 'Z2', 'X2']
    ),
    'exp7': dict(
        Z3=[],
        X2=['Z3'],
        X=['X2', ('X2',)],
        Z1=['X', ('Z3',)],
        Z2=['Z3', ('X2',)],
        Y=['Z1', 'Z2', 'X2']
    ),
    'exp8': dict(
        W=[],
        X=['W', ('W',)],
        Z=['W', 'X'],
        R=['W'],
        Y=['Z', 'R', ('W',), ('X',)]
    ),
    # ---- ch, cc, d models reordered for correct sampling ----
    '5-ch': dict(
        X=[],
        V1=['X'],
        V2=['V1', ('X',)],
        V3=['V2'],
        Y=['V3', ('V2',)]
    ),
    '6-cc': dict(
        V3=[],
        V4=[('V3',)],
        X=[('V4',)],
        V1=['V3', 'V4'],
        V2=['V4', 'X'],
        Y=['V1', 'V2', ('V3',), ('X',)]
    ),
    '9-ch': dict(
        X=[],
        V1=['X'],
        V2=['V1', ('X',)],
        V3=['V2'],
        V4=['V3', ('V2',)],
        V5=['V4'],
        V6=['V5', ('V4',)],
        V7=['V6'],
        Y=['V7', ('V6',)]
    ),
    '9-d': dict(
        X=[],
        V1=['X'],
        V2=['V1', ('X',)],
        V3=['X'],
        V4=['V3', ('X',)],
        V5=['X'],
        V6=['V5', ('X',)],
        V7=['X'],
        Y=['V7', 'V2', 'V4', 'V6', ('X',)]
    ),
    '15-cc': dict(
        V10=[],
        V11=[('V10',)],
        V12=[('V11',)],
        V13=[('V12',)],
        X=[('V13',)],
        V6=['V10', 'V11'],
        V7=['V11', 'V12'],
        V3=['V6', 'V7', ('V10',)],
        V8=['V12', 'V13'],
        V4=['V7', 'V8'],
        V1=['V3', 'V4'],
        V9=['V13', 'X'],
        V5=['V8', 'V9', ('X',)],
        V2=['V4', 'V5'],
        Y=['V1', 'V2', ('V3',), ('V5',)]
    ),
    '17-d': dict(
        X=[],
        V1=['X'],
        V2=['V1', ('X',)],
        V3=['V2'],
        V4=['V3', ('V2',)],
        V5=['X'],
        V6=['V5', ('X',)],
        V7=['V6'],
        V8=['V7', ('V6',)],
        V9=['X'],
        V10=['V9', ('X',)],
        V11=['V10'],
        V12=['V11', ('V10',)],
        V13=['X'],
        V14=['V13', ('X',)],
        V15=['V14'],
        Y=['V15', 'V4', 'V8', 'V12', ('V14',)]
    ),
    '25-ch': dict(
        X=[],
        V1=['X'],
        V2=['V1', ('X',)],
        V3=['V2'],
        V4=['V3', ('V2',)],
        V5=['V4'],
        V6=['V5', ('V4',)],
        V7=['V6'],
        V8=['V7', ('V6',)],
        V9=['V8'],
        V10=['V9', ('V8',)],
        V11=['V10'],
        V12=['V11', ('V10',)],
        V13=['V12'],
        V14=['V13', ('V12',)],
        V15=['V14'],
        V16=['V15', ('V14',)],
        V17=['V16'],
        V18=['V17', ('V16',)],
        V19=['V18'],
        V20=['V19', ('V18',)],
        V21=['V20'],
        V22=['V21', ('V20',)],
        V23=['V22'],
        Y=['V23', ('V22',)]
    ),
    '45-cc': dict(
        V36=[],
        V37=[('V36',)],
        V38=[('V37',)],
        V39=[('V38',)],
        V40=[('V39',)],
        V41=[('V40',)],
        V42=[('V41',)],
        V43=[('V42',)],
        V44=[('V43',)],
        V28=['V36', 'V37'],
        V29=['V37', 'V38'],
        V21=['V28', 'V29', ('V36',)],
        V30=['V38', 'V39'],
        V22=['V29', 'V30'],
        V15=['V21', 'V22'],
        V31=['V39', 'V40'],
        V23=['V30', 'V31'],
        V16=['V22', 'V23'],
        V10=['V15', 'V16', ('V21',)],
        V32=['V40', 'V41'],
        V24=['V31', 'V32'],
        V17=['V23', 'V24'],
        V11=['V16', 'V17'],
        V6=['V10', 'V11'],
        V33=['V41', 'V42'],
        V25=['V32', 'V33'],
        V18=['V24', 'V25'],
        V12=['V17', 'V18'],
        V7=['V11', 'V12'],
        V3=['V6', 'V7', ('V10',)],
        V34=['V42', 'V43'],
        V26=['V33', 'V34'],
        V19=['V25', 'V26'],
        V13=['V18', 'V19'],
        V8=['V12', 'V13'],
        V4=['V7', 'V8'],
        V1=['V3', 'V4'],
        V35=['V43', 'V44'],
        V27=['V34', 'V35', ('V44',)],
        V20=['V26', 'V27'],
        V14=['V19', 'V20', ('V27',)],
        V9=['V13', 'V14'],
        V5=['V8', 'V9', ('V14',)],
        V2=['V4', 'V5'],
        Y=['V1', 'V2', ('V3',), ('V5',)]
    ),
    '49-ch': dict(
        X=[],
        V1=['X'],
        V2=['V1', ('X',)],
        V3=['V2'],
        V4=['V3', ('V2',)],
        V5=['V4'],
        V6=['V5', ('V4',)],
        V7=['V6'],
        V8=['V7', ('V6',)],
        V9=['V8'],
        V10=['V9', ('V8',)],
        V11=['V10'],
        V12=['V11', ('V10',)],
        V13=['V12'],
        V14=['V13', ('V12',)],
        V15=['V14'],
        V16=['V15', ('V14',)],
        V17=['V16'],
        V18=['V17', ('V16',)],
        V19=['V18'],
        V20=['V19', ('V18',)],
        V21=['V20'],
        V22=['V21', ('V20',)],
        V23=['V22'],
        V24=['V23', ('V22',)],
        V25=['V24'],
        V26=['V25', ('V24',)],
        V27=['V26'],
        V28=['V27', ('V26',)],
        V29=['V28'],
        V30=['V29', ('V28',)],
        V31=['V30'],
        V32=['V31', ('V30',)],
        V33=['V32'],
        V34=['V33', ('V32',)],
        V35=['V34'],
        V36=['V35', ('V34',)],
        V37=['V36'],
        V38=['V37', ('V36',)],
        V39=['V38'],
        V40=['V39', ('V38',)],
        V41=['V40'],
        V42=['V41', ('V40',)],
        V43=['V42'],
        V44=['V43', ('V42',)],
        V45=['V44'],
        V46=['V45', ('V44',)],
        V47=['V46'],
        Y=['V47', ('V46',)]
    ),
    '65-d': dict(
        X=[],
        V1=['X'],
        V2=['V1', ('X',)],
        V3=['V2'],
        V4=['V3', ('V2',)],
        V5=['V4'],
        V6=['V5', ('V4',)],
        V7=['V6'],
        V8=['V7', ('V6',)],
        V9=['V8'],
        V10=['V9', ('V8',)],
        V11=['V10'],
        V12=['V11', ('V10',)],
        V13=['V12'],
        V14=['V13', ('V12',)],
        V15=['V14'],
        V16=['V15', ('V14',)],
        V17=['X'],
        V18=['V17', ('X',)],
        V19=['V18'],
        V20=['V19', ('V18',)],
        V21=['V20'],
        V22=['V21', ('V20',)],
        V23=['V22'],
        V24=['V23', ('V22',)],
        V25=['V24'],
        V26=['V25', ('V24',)],
        V27=['V26'],
        V28=['V27', ('V26',)],
        V29=['V28'],
        V30=['V29', ('V28',)],
        V31=['V30'],
        V32=['V31', ('V30',)],
        V33=['X'],
        V34=['V33', ('X',)],
        V35=['V34'],
        V36=['V35', ('V34',)],
        V37=['V36'],
        V38=['V37', ('V36',)],
        V39=['V38'],
        V40=['V39', ('V38',)],
        V41=['V40'],
        V42=['V41', ('V40',)],
        V43=['V42'],
        V44=['V43', ('V42',)],
        V45=['V44'],
        V46=['V45', ('V44',)],
        V47=['V46'],
        V48=['V47', ('V46',)],
        V49=['X'],
        V50=['V49', ('X',)],
        V51=['V50'],
        V52=['V51', ('V50',)],
        V53=['V52'],
        V54=['V53', ('V52',)],
        V55=['V54'],
        V56=['V55', ('V54',)],
        V57=['V56'],
        V58=['V57', ('V56',)],
        V59=['V58'],
        V60=['V59', ('V58',)],
        V61=['V60'],
        V62=['V61', ('V60',)],
        V63=['V62'],
        Y=['V63', 'V16', 'V32', 'V48', ('V62',)]
    ),
    '99-ch': dict(
        X=[],
        V1=['X'],
        V2=['V1', ('X',)],
        V3=['V2'],
        V4=['V3', ('V2',)],
        V5=['V4'],
        V6=['V5', ('V4',)],
        V7=['V6'],
        V8=['V7', ('V6',)],
        V9=['V8'],
        V10=['V9', ('V8',)],
        V11=['V10'],
        V12=['V11', ('V10',)],
        V13=['V12'],
        V14=['V13', ('V12',)],
        V15=['V14'],
        V16=['V15', ('V14',)],
        V17=['V16'],
        V18=['V17', ('V16',)],
        V19=['V18'],
        V20=['V19', ('V18',)],
        V21=['V20'],
        V22=['V21', ('V20',)],
        V23=['V22'],
        V24=['V23', ('V22',)],
        V25=['V24'],
        V26=['V25', ('V24',)],
        V27=['V26'],
        V28=['V27', ('V26',)],
        V29=['V28'],
        V30=['V29', ('V28',)],
        V31=['V30'],
        V32=['V31', ('V30',)],
        V33=['V32'],
        V34=['V33', ('V32',)],
        V35=['V34'],
        V36=['V35', ('V34',)],
        V37=['V36'],
        V38=['V37', ('V36',)],
        V39=['V38'],
        V40=['V39', ('V38',)],
        V41=['V40'],
        V42=['V41', ('V40',)],
        V43=['V42'],
        V44=['V43', ('V42',)],
        V45=['V44'],
        V46=['V45', ('V44',)],
        V47=['V46'],
        V48=['V47', ('V46',)],
        V49=['V48'],
        V50=['V49', ('V48',)],
        V51=['V50'],
        V52=['V51', ('V50',)],
        V53=['V52'],
        V54=['V53', ('V52',)],
        V55=['V54'],
        V56=['V55', ('V54',)],
        V57=['V56'],
        V58=['V57', ('V56',)],
        V59=['V58'],
        V60=['V59', ('V58',)],
        V61=['V60'],
        V62=['V61', ('V60',)],
        V63=['V62'],
        V64=['V63', ('V62',)],
        V65=['V64'],
        V66=['V65', ('V64',)],
        V67=['V66'],
        V68=['V67', ('V66',)],
        V69=['V68'],
        V70=['V69', ('V68',)],
        V71=['V70'],
        V72=['V71', ('V70',)],
        V73=['V72'],
        V74=['V73', ('V72',)],
        V75=['V74'],
        V76=['V75', ('V74',)],
        V77=['V76'],
        V78=['V77', ('V76',)],
        V79=['V78'],
        V80=['V79', ('V78',)],
        V81=['V80'],
        V82=['V81', ('V80',)],
        V83=['V82'],
        V84=['V83', ('V82',)],
        V85=['V84'],
        V86=['V85', ('V84',)],
        V87=['V86'],
        V88=['V87', ('V86',)],
        V89=['V88'],
        V90=['V89', ('V88',)],
        V91=['V90'],
        V92=['V91', ('V90',)],
        V93=['V92'],
        V94=['V93', ('V92',)],
        V95=['V94'],
        V96=['V95', ('V94',)],
        V97=['V96'],
        Y=['V97', ('V96',)]
    )
}


class CTM(SCM):
    def __init__(self, cg_file=None, rpa=None, v_size={}):
        assert (cg_file is None) != (rpa is None)
        if cg_file is not None:
            name = cg_file.split('/')[-1].split('.')[0]
            if name not in RPA:
                raise ValueError(f"Graph '{name}' is unsupported")
            self.rpa = RPA[name]
        else:
            self.rpa = rpa
        self.r = {k: ((k, ()),) if not self.rpa[k] else tuple(sorted(
            (k, vals)
            for vals in itertools.product(*(
                [(k2, 0), (k2, 1)]
                for k2 in self.rpa[k]
                if type(k2) is str
            )))) for k in self.rpa}
        self.cond = {self.r[k]: list(itertools.chain.from_iterable(
            self.r[k2[0]] for k2 in self.rpa[k] if type(k2) is tuple))
            for k in self.rpa}

        v = list(self.r)
        pu = FactorizedDistribution(self.r.values(), cond=self.cond)
        f = {vi: lambda v, u, vi=vi, r=self.r, rpa=self.rpa:
             (T.cat([u[ui] for ui in self.r[vi]], dim=-1)
              .reshape((u[self.r[vi][0]].shape[0],)
                       + (2,) * len(self.r[vi][0][1]))[
                  (T.arange(u[self.r[vi][0]].shape[0]),)
                  + tuple(
                      (v[k] if isinstance(k, str) else u[k]).flatten()
                      for k, _ in self.r[vi][0][1]
                  )]).view(-1, 1) for vi in v}

        super().__init__(v, f, pu)

    def pmf(self, v, do={}, cond={}):
        pmf = T.exp(self.log_pmf(v, do, cond))
        return pmf if self.training else pmf.item()
        if cond:
            assert set(v).isdisjoint(cond)
            return (self.pmf(dict(v, **cond), do=do)
                    / self.pmf(cond, do=do))

        def _compare(v1, v2):
            return (all(v1[k] == v2[k].item() for k in v1))
        pmf = sum((T.exp(self.pu.log_pmf(u))
                   for u in self.pu.space()
                   if _compare(v, self(u=u, do=do))),
                  T.tensor(0.))
        return pmf if self.training else pmf.item()

    def log_pmf(self, v, do={}, cond={}):
        if cond:
            assert set(v).isdisjoint(cond)
            return (self.log_pmf(dict(v, **cond), do=do)
                    - self.log_pmf(cond, do=do))

        def _compare(v1, v2):
            return (all(v1[k] == v2[k].item() for k in v1))
        pmf = T.cat([self.pu.log_pmf(u)
                     for u in self.pu.space()
                     if _compare(v, self(u=u, do=do))], dim=-1)
        return T.logsumexp(pmf, dim=-1, keepdim=True)
