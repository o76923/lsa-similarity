from os import getenv
from py.create_space import Creator
from py.calculate_sim import LSASim

space_name = getenv("SPACE")

try:
    paragraph = getenv("PARAGRAPH")
    c = Creator(paragraph, space_name)
    c.main()
except KeyError:
    pass

try:
    target = getenv("TARGET")
    s = LSASim(target, space_name)
    s.main()
except KeyError:
    pass
