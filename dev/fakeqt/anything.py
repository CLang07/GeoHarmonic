class _Meta(type):
    def __getattr__(cls, n): return Anything()
class Anything(metaclass=_Meta):
    def __init__(self,*a,**k): pass
    def __getattr__(self,n):
        if n.startswith('__'): raise AttributeError(n)
        return Anything()
    def __call__(self,*a,**k): return Anything()
    def __iter__(self): return iter([])
    def __bool__(self): return False
    def __and__(self,o): return Anything()
    __or__=__rand__=__ror__=__and__
    def __int__(self): return 0
    def __float__(self): return 0.0
    def __len__(self): return 0
    def __getitem__(self,k): return Anything()
for _n in ('__lt__','__gt__','__le__','__ge__'):
    setattr(Anything,_n,lambda s,o: False)
Anything.__index__=lambda s:0
Anything.__invert__=lambda s: Anything()
