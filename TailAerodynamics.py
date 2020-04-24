import openmdao.api as om
import numpy as np


class AspectRatioTail(om.ExplicitComponent):

    def setup(self):

        self.add_input('tailSpan')
        self.add_input('tailChord')

        self.add_output('tailAR')

        self.declare_partials('*', '*', method='cs')

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        b_t = inputs['tailSpan']
        c_t = inputs['tailChord']

        AR_t = b_t / c_t

        outputs['tailAR'] = AR_t


class LiftCoefficientTail(om.ExplicitComponent):

    def setup(self):

        self.add_input('alphaT', units='rad')
        self.add_input('tailAR')
        self.add_input('tailSpanwiseEfficiency')

        self.add_output('tailCL')
        self.add_output('taildCLdAlpha', units='rad**-1')

        self.declare_partials(['tailCL'], ['alphaT', 'tailAR'])
        self.declare_partials('*', '*', method='cs')

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        e_t = inputs['tailSpanwiseEfficiency']
        AR_t = inputs['tailAR']
        alpha_t = inputs['alphaT']

        CL_t = 2 * np.pi * ((AR_t * e_t)/ ((AR_t*e_t) + 2)) * alpha_t

        outputs['taildCLdAlpha'] = 2*np.pi*(AR_t/(AR_t+2))
        outputs['tailCL'] = CL_t


class TailEffectiveAngle(om.ExplicitComponent):

    def setup(self):

        self.add_input('dEpsilon_dAlpha', units='rad**-1')
        self.add_input('alpha', units='rad')
        self.add_input('alphaZeroLift', units='rad')
        self.add_input('tailAngle', units='rad')
        self.add_input('tailCL')
        self.add_input('tailAR')
        self.add_input('tailSpanwiseEfficiency')

        self.add_output('alphaT', units='rad')

        self.declare_partials('*', '*', method='cs')

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        epsilon_a = inputs['dEpsilon_dAlpha']
        alpha = inputs['alpha']
        alpha_0 = inputs['alphaZeroLift']
        alpha_s = inputs['tailAngle']
        CL_t = inputs['tailCL']
        AR_t = inputs['tailAR']
        e_t = inputs['tailSpanwiseEfficiency']

        alpha_T = (1-epsilon_a)*alpha + epsilon_a*alpha_0 + alpha_s - (CL_t)/(np.pi*AR_t*e_t)

        outputs['alphaT'] = alpha_T


class InducedDragCoefficientTail(om.ExplicitComponent):

    def setup(self):

        self.add_input('tailCL')
        self.add_input('tailAR')
        self.add_input('tailSpanwiseEfficiency')

        self.add_output('tailCDi')

        self.declare_partials('*', '*', method='cs')

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        CL_t = inputs['tailCL']
        AR_t = inputs['tailAR']
        e_t = inputs['tailSpanwiseEfficiency']

        CDi_t = (CL_t**2)/(np.pi*AR_t*e_t)

        outputs['tailCDi'] = CDi_t


class TailLift(om.ExplicitComponent):

    def setup(self):

        self.add_input('tailCL')
        self.add_input('airspeed', units='m/s')
        self.add_input('airDensity', units='kg/m**3')
        self.add_input('tailSpan', units='m')
        self.add_input('tailChord', units='m')

        self.add_output('tailLift', units='N')

        self.declare_partials('*', '*', method='cs')

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        CL_t = inputs['tailCL']
        V = inputs['airspeed']
        rho = inputs['airDensity']
        b_t = inputs['tailSpan']
        c_t = inputs['tailChord']

        L_t = 0.5*rho*(V**2)*b_t*c_t*CL_t

        outputs['tailLift'] = L_t


class TailInducedDrag(om.ExplicitComponent):

    def setup(self):

        self.add_input('tailCDi')
        self.add_input('airspeed', units='m/s')
        self.add_input('airDensity', units='kg/m**3')
        self.add_input('tailSpan', units='m')
        self.add_input('tailChord', units='m')

        self.add_output('tailInducedDrag', units='N')

        self.declare_partials('*', '*', method='cs')

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        CDi_t = inputs['tailCDi']
        V = inputs['airspeed']
        rho = inputs['airDensity']
        b_t = inputs['tailSpan']
        c_t = inputs['tailChord']

        Di_t = 0.5 * rho * (V ** 2) * b_t * c_t * CDi_t

        outputs['tailInducedDrag'] = Di_t


class TailSkinDrag(om.ExplicitComponent):

    def setup(self):

        self.add_input('tailChord', units='m')
        self.add_input('tailSpan', units='m')
        self.add_input('airspeed', units='m/s')
        self.add_input('airDensity', units='kg/m**3')
        self.add_input('airViscosity', units='kg/(m*s)')

        self.add_output('tailSkinDrag', units='N')

        self.declare_partials('*', '*', method='cs')

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        c_t = inputs['tailChord']
        b_t = inputs['tailSpan']
        V = inputs['airspeed']
        rho = inputs['airDensity']
        mu = inputs['airViscosity']

        Df_t = 2*((63*(c_t**(6/7))*(rho**(6/7))*(mu**(1/7))*(V**(13/7)))/4000)*b_t

        outputs['tailSkinDrag'] = Df_t


class TailGroup(om.Group):

    def setup(self):

        self.add_subsystem('tailCL_sys', LiftCoefficientTail(),
                           promotes=['*'])
        self.add_subsystem('tailAlphaEff_sys', TailEffectiveAngle(),
                           promotes=['*'])
        self.add_subsystem('tailCDi_sys', InducedDragCoefficientTail(),
                           promotes=['*'])
        self.add_subsystem('tailLift_sys', TailLift(),
                           promotes=['*'])
        self.add_subsystem('tailInducedDrag_sys', TailInducedDrag(),
                           promotes=['*'])
        self.add_subsystem('tailSkinDrag_sys', TailSkinDrag(),
                           promotes=['*'])

        self.nonlinear_solver = om.NewtonSolver(solve_subsystems=True)
        self.nonlinear_solver.linesearch = om.ArmijoGoldsteinLS()
        self.linear_solver = om.DirectSolver()