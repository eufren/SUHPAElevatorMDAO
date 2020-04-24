import openmdao.api as om
import numpy as np


class WingDownwash(om.ExplicitComponent):

    def setup(self):

        self.add_input('dCL_dAlpha', units='rad**-1')
        self.add_input('wingAR')
        self.add_input('wingSpanwiseEfficiency')

        self.add_output('dEpsilon_dAlpha', units='rad**-1')

        self.declare_partials('*', '*', method='cs')

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        CL_a = inputs['dCL_dAlpha']
        AR = inputs['wingAR']
        e = inputs['wingSpanwiseEfficiency']

        e_a = CL_a/(np.pi*AR*e)

        outputs['dEpsilon_dAlpha'] = e_a


class LiftCoefficientWing(om.ExplicitComponent):

    def setup(self):

        self.add_input('dCL_dAlpha', units='rad**-1')
        self.add_input('alpha', units='rad')
        self.add_input('wingSettingAngle', units='rad')
        self.add_input('alphaZeroLift', units='rad')

        self.add_output('wingCL')

        self.declare_partials('*', '*', method='cs')

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        CL_a = inputs['dCL_dAlpha']
        alpha = inputs['alpha']
        alpha_w = inputs['wingSettingAngle']
        alpha_0 = inputs['alphaZeroLift']

        CL = CL_a*(alpha-alpha_w-alpha_0)

        outputs['wingCL'] = CL


class DragCoefficientWing(om.ExplicitComponent):

    def setup(self):

        self.add_input('wingCL')
        self.add_input('wingAR')
        self.add_input('wingSpanwiseEfficiency')
        self.add_input('wingCD_Min')
        self.add_input('wingCL_DMin')

        self.add_output('wingCD')

        self.declare_partials('*', '*', method='cs')

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        CL = inputs['wingCL']
        AR = inputs['wingAR']
        e = inputs['wingSpanwiseEfficiency']
        CD_Min = inputs['wingCD_Min']
        CL_DMin = inputs['wingCL_DMin']

        K = 1/(np.pi*AR*e)

        CD = CD_Min + K*(CL-CL_DMin)**2

        outputs['wingCD'] = CD


class WingLift(om.ExplicitComponent):

    def setup(self):

        self.add_input('wingCL')
        self.add_input('airspeed', units='m/s')
        self.add_input('airDensity', units='kg/m**3')
        self.add_input('wingArea', units='m**2')

        self.add_output('wingLift', units='N')

        self.declare_partials('*', '*', method='cs')

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        CL = inputs['wingCL']
        V = inputs['airspeed']
        rho = inputs['airDensity']
        S = inputs['wingArea']

        L = 0.5 * rho * (V ** 2) * S * CL

        outputs['wingLift'] = L


class WingDrag(om.ExplicitComponent):

    def setup(self):

        self.add_input('wingCD')
        self.add_input('airspeed', units='m/s')
        self.add_input('airDensity', units='kg/m**3')
        self.add_input('wingArea', units='m**2')

        self.add_output('wingDrag', units='N')

        self.declare_partials('*', '*', method='cs')

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        CD = inputs['wingCD']
        V = inputs['airspeed']
        rho = inputs['airDensity']
        S = inputs['wingArea']

        D = 0.5 * rho * (V ** 2) * S * CD

        outputs['wingDrag'] = D


class WingMoment(om.ExplicitComponent):

    def setup(self):

        self.add_input('wingCM')
        self.add_input('airspeed', units='m/s')
        self.add_input('airDensity', units='kg/m**3')
        self.add_input('wingArea', units='m**2')
        self.add_input('wingMAC', units='m')

        self.add_output('wingMoment', units='N*m')

        self.declare_partials('*', '*', method='cs')

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        CM = inputs['wingCM']
        V = inputs['airspeed']
        rho = inputs['airDensity']
        S = inputs['wingArea']
        c = inputs['wingMAC']

        M = 0.5 * rho * (V ** 2) * S * c * CM

        outputs['wingMoment'] = M


class WingGroup(om.Group):

    def setup(self):

        self.add_subsystem('wingLiftCoefficient_sys', LiftCoefficientWing(),
                           promotes=['*'])
        self.add_subsystem('wingDragCoefficient_sys', DragCoefficientWing(),
                           promotes=['*'])
        self.add_subsystem('wingDownwash_sys', WingDownwash(),
                           promotes=['*'])
        self.add_subsystem('wingLift_sys', WingLift(),
                           promotes=['*'])
        self.add_subsystem('wingDrag_sys', WingDrag(),
                           promotes=['*'])
        self.add_subsystem('wingMoment_sys', WingMoment(),
                           promotes=['*'])
