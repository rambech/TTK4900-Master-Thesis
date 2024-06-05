from test.tests import *
from test.direct_collocation import *
import time

# new_distance_example()
# otter_distance_example()
# dubins_distance_example()
# test_mpc()
# test_mpc_simulator()
# test_ravnkloa()
# test_brattora()
# test_nidelva()
# test_simulator()
# direct_collocation_example()
# opti_direct_collocation_example()
# as_direct_collocation_example()
# mpc_direct_collocation_example()
# otter_mpc_direct_collocation_example()
# otter_direct_collocation_example()
# test_gradient()

# # Simple Brattøra
# test_brattora(rl=False, default=True,
#               estimate_current=False,
#               V_c=0, simple=True, B=1)

# test_brattora(rl=True, default=False,
#               estimate_current=True,
#               V_c=0, simple=True, B=1, sysid=False)

# test_brattora(rl=True, default=False,
#               estimate_current=True,
#               V_c=0, simple=True, B=1)

# test_brattora(rl=True, default=False,
#               estimate_current=True,
#               V_c=0, simple=True, B=10)

# # Otter Brattøra
# # NMPC no current
# test_brattora(rl=False, default=True,
#               estimate_current=False,
#               V_c=0, simple=False)

# NMPC current
test_brattora(rl=False, default=True,
              estimate_current=False,
              V_c=utils.kts2ms(1),
              simple=False)

# # RL no current
# test_brattora(rl=True, default=True,
#               estimate_current=True,
#               V_c=0, simple=False, B=1)

# test_brattora(rl=True, default=False,
#               estimate_current=True,
#               V_c=0, simple=False, B=1)

# test_brattora(rl=True, default=False,
#               estimate_current=True,
#               V_c=0, simple=False, B=10)

# # RL current
# test_brattora(rl=True, default=True,
#               estimate_current=True,
#               V_c=utils.kts2ms(1),
#               simple=False, B=1)

# test_brattora(rl=True, default=False,
#               estimate_current=True,
#               V_c=utils.kts2ms(1), simple=False, B=1)

# test_brattora(rl=True, default=False,
#               estimate_current=True,
#               V_c=utils.kts2ms(1),
#               simple=False, B=10)

# # Simple Ravnkloa
# test_ravnkloa(rl=False, default=True,
#               estimate_current=False,
#               V_c=0, simple=True, B=1)

# test_ravnkloa(rl=True, default=False,
#               estimate_current=True,
#               V_c=0, simple=True, B=1, sysid=False)

# test_ravnkloa(rl=True, default=False,
#               estimate_current=True,
#               V_c=0, simple=True, B=1)

# test_ravnkloa(rl=True, default=False,
#               estimate_current=True,
#               V_c=0, simple=True, B=10)

# # Otter Ravnkloa
# # NMPC no current
# test_ravnkloa(rl=False, default=True,
#               estimate_current=False,
#               V_c=0, simple=False)

# # NMPC current
# test_ravnkloa(rl=False, default=True,
#               estimate_current=False,
#               V_c=utils.kts2ms(1.5),
#               simple=False)

# # RL no current
# test_ravnkloa(rl=True, default=True,
#               estimate_current=True,
#               V_c=0, simple=False, B=1)

# test_ravnkloa(rl=True, default=False,
#               estimate_current=True,
#               V_c=0, simple=False, B=1)

# test_ravnkloa(rl=True, default=False,
#               estimate_current=True,
#               V_c=0, simple=False, B=10)

# # RL current
# test_ravnkloa(rl=True, default=True,
#               estimate_current=True,
#               V_c=utils.kts2ms(1.5),
#               simple=False, B=1)

# test_ravnkloa(rl=True, default=False,
#               estimate_current=True,
#               V_c=utils.kts2ms(1.5), simple=False, B=1)

# test_ravnkloa(rl=True, default=False,
#               estimate_current=True,
#               V_c=utils.kts2ms(1.5),
#               simple=False, B=10)

# Simple Nidelva
# test_nidelva(rl=False, default=True,
#              estimate_current=False,
#              V_c=0, simple=True, B=1)

# test_nidelva(rl=True, default=False,
#              estimate_current=True,
#              V_c=0, simple=True, B=1, sysid=False)

# test_nidelva(rl=True, default=False,
#              estimate_current=True,
#              V_c=0, simple=True, B=1)

# test_nidelva(rl=True, default=False,
#              estimate_current=True,
#              V_c=0, simple=True, B=10)

# # Otter Nidelva
# # NMPC no current
# test_nidelva(rl=False, default=True,
#              estimate_current=False,
#              V_c=0, simple=False)

# # NMPC current
test_nidelva(rl=False, default=True,
             estimate_current=False,
             V_c=utils.kts2ms(1),
             simple=False)

# # RL no current
# test_nidelva(rl=True, default=True,
#              estimate_current=True,
#              V_c=0, simple=False, B=1)

# test_nidelva(rl=True, default=False,
#              estimate_current=True,
#              V_c=0, simple=False, B=1)

# test_nidelva(rl=True, default=False,
#              estimate_current=True,
#              V_c=0, simple=False, B=10)

# # RL current
# test_nidelva(rl=True, default=True,
#              estimate_current=True,
#              V_c=utils.kts2ms(1),
#              simple=False, B=1)

# test_nidelva(rl=True, default=False,
#              estimate_current=True,
#              V_c=utils.kts2ms(1), simple=False, B=1)

# test_nidelva(rl=True, default=False,
#              estimate_current=True,
#              V_c=utils.kts2ms(1),
#              simple=False, B=10)
