import openmdao.api as om
import scipy.constants as scon
from TailAerodynamics import TailGroup
from WingAerodynamics import WingGroup
from Weights import WeightsGroup
from Constraints import TotalDrag, TotalVerticalForce, TotalMomentAboutCG, TailVolume, StaticMargin


class ElevatorModel(om.Group):

    def setup(self):

        # Design variables
        designVars = self.add_subsystem('designVariables', om.IndepVarComp(), promotes=['*'])
        designVars.add_output('alpha', val=3, units='deg')
        designVars.add_output('tailAngle', val=-3, units='deg')
        designVars.add_output('tailChord', val=0.42, units='m')
        designVars.add_output('tailSpan', val=3, units='m')
        designVars.add_output('tailDistance', val=3.5, units='m')
        designVars.add_output('wingDistance', val=0.2, units='m')

        # Wing parameters
        wingParams = self.add_subsystem('wingParameters', om.IndepVarComp(), promotes=['*'])
        wingParams.add_output('wingArea', val=20.4, units='m**2')
        wingParams.add_output('wingSpan', val=24, units='m')
        wingParams.add_output('wingAR', val=28.235)
        wingParams.add_output('wingSpanwiseEfficiency', val=0.97)
        wingParams.add_output('wingMAC', val=0.894, units='m')
        wingParams.add_output('dCL_dAlpha', val=0.102, units='deg**-1')
        wingParams.add_output('wingCM', val=-0.222)
        wingParams.add_output('wingACDistance', val=0.255, units='m')
        wingParams.add_output('wingMass', val=14, units='kg')
        wingParams.add_output('alphaZeroLift', val=-7.48, units='deg')
        wingParams.add_output('wingCD_Min', val=0.018)
        wingParams.add_output('wingCL_DMin', val=0.55)

        # Tail parameters
        tailParams = self.add_subsystem('tailParameters', om.IndepVarComp(), promotes=['*'])
        tailParams.add_output('tailSpanwiseEfficiency', val=0.8)
        tailParams.add_output('tailMassPerArea', val=0.6, units='kg/m**2')

        # Aircraft parameters
        acParams = self.add_subsystem('aircraftParameters', om.IndepVarComp(), promotes=['*'])
        acParams.add_output('restOfAircraftCGDistance', val=0.711, units='m')
        acParams.add_output('restOfAircraftMass', val=82.7, units='kg')
        acParams.add_output('boomMassPerLength', val=1, units='kg/m')

        # Environmental constants
        envConsts = self.add_subsystem('envConstants', om.IndepVarComp(), promotes=['*'])
        envConsts.add_output('g', val=scon.g, units='m/s**2')
        envConsts.add_output('airspeed', val=8.25, units='m/s')
        envConsts.add_output('airDensity', val=1.225, units='kg/m**3')
        envConsts.add_output('airViscosity', val=1.802E-5, units='kg/(m*s)')

        # Analysis components
        self.add_subsystem('wingGroup', WingGroup(),
                           promotes_inputs=['alpha', 'dCL_dAlpha', 'wingAR', 'wingSpanwiseEfficiency', 'alphaZeroLift',
                                            'wingCD_Min', 'wingCM', 'airspeed', 'airDensity',
                                            'wingArea', 'wingMAC', 'wingCL_DMin'],
                           promotes_outputs=['dEpsilon_dAlpha', 'wingLift', 'wingDrag', 'wingMoment'])


        self.add_subsystem('tailGroup', TailGroup(),
                           promotes_inputs=['alpha', 'tailChord', 'tailSpan', 'airspeed', 'airDensity', 'airViscosity',
                                            'tailSpanwiseEfficiency', 'dEpsilon_dAlpha', 'tailAngle', 'alphaZeroLift'],
                           promotes_outputs=['tailLift', 'tailInducedDrag', 'tailSkinDrag', 'taildCLdAlpha'])


        self.add_subsystem('weightsGroup', WeightsGroup(),
                           promotes_inputs=['tailSpan', 'tailChord', 'tailMassPerArea', 'g', 'tailDistance',
                                            'wingDistance', 'wingACDistance', 'boomMassPerLength',
                                            'restOfAircraftCGDistance', 'restOfAircraftMass', 'wingMass'],
                           promotes_outputs=['tailWeight', 'boomWeight', 'totalCGDistance'])

        # Constraint systems
        self.add_subsystem('totalVerticalForce_sys', TotalVerticalForce(),
                           promotes=['*'])

        self.add_subsystem('totalVDrag_sys', TotalDrag(),
                           promotes=['*'])

        self.add_subsystem('totalMomentAboutCG_sys', TotalMomentAboutCG(),
                           promotes=['*'])

        self.add_subsystem('tailVolume_sys', TailVolume(),
                           promotes=['*'])

        self.add_subsystem('statMargin_sys', StaticMargin(),
                           promotes=['*'])

# Create optimisation problem
prob = om.Problem()
prob.model = ElevatorModel()

# Configure problem driver
prob.driver = om.ScipyOptimizeDriver()
prob.driver.options['optimizer'] = 'SLSQP'
prob.driver.options['maxiter'] = 100
prob.driver.options['tol'] = 1e-5

# Set information printing to
prob.driver.options['debug_print'] = ['desvars', 'ln_cons', 'nl_cons', 'objs']

# Add design variables, constraints and objective
prob.model.add_design_var('wingDistance', lower=0, upper=0.4)
prob.model.add_design_var('tailChord', lower=0.2, upper=1)
prob.model.add_design_var('tailSpan', lower=0.2, upper=5)
prob.model.add_design_var('tailDistance', lower=3, upper=6)
prob.model.add_design_var('alpha', lower=-4, upper=6)
prob.model.add_design_var('tailAngle', lower=-8, upper=8)

prob.model.add_constraint('verticalForce', equals=0)
prob.model.add_constraint('momentAboutCG', equals=0)
prob.model.add_constraint('staticMargin', equals=3)

prob.model.add_objective('totalDrag')

prob.setup()

#prob.check_config()
prob.check_partials(compact_print=True, show_only_incorrect=True)
#prob.model.approx_totals()
#prob.run_driver()
