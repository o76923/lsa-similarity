from os import getenv
from py.create_space import Creator
from py.calculate_sim import LSASim

space_name = getenv("SPACE")

paragraph = getenv("PARAGRAPH", None)
if paragraph:
    c = Creator(paragraph, space_name)
    c.main()

target = getenv("TARGET", None)
if target:
    s = LSASim(target, space_name)
    s.main()
