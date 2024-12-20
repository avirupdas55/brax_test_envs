from brax.envs import register_environment
from .unit_test_envs import *

unit_envs = {
    'first': FirstEnv,
    'second': SecondEnv,
    'third': ThirdEnv,
    'fourth': FourthEnv,
    'fifth': FifthEnv,
}

for name, env in unit_envs.items():
    register_environment(name, env)