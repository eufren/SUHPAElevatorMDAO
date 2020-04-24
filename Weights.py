import openmdao.api as om


class TailMassAndWeight(om.ExplicitComponent):

    def setup(self):

        self.add_input('tailSpan', units='m')
        self.add_input('tailChord', units='m')
        self.add_input('tailMassPerArea', units='kg/m**2')
        self.add_input('g', units='m/s**2')

        self.add_output('tailWeight', units='N')
        self.add_output('tailMass', units='kg')

        self.declare_partials(['tailMass'], ['tailSpan', 'tailChord', 'tailMassPerArea'])
        self.declare_partials(['tailWeight'], ['tailSpan', 'tailChord', 'tailMassPerArea'])

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        b_t = inputs['tailSpan']
        c_t = inputs['tailChord']
        rho_t = inputs['tailMassPerArea']
        g = inputs['g']

        m_t = b_t * c_t * rho_t
        W_t = m_t * g

        outputs['tailMass'] = m_t
        outputs['tailWeight'] = W_t

    def compute_partials(self, inputs, partials, discrete_inputs=None):

        b_t = inputs['tailSpan']
        c_t = inputs['tailChord']
        rho_t = inputs['tailMassPerArea']
        g = inputs['g']

        partials['tailMass', 'tailSpan'] = c_t * rho_t
        partials['tailMass', 'tailChord'] = b_t * rho_t
        partials['tailMass', 'tailMassPerArea'] = c_t * b_t

        partials['tailWeight', 'tailSpan'] = c_t * rho_t * g
        partials['tailWeight', 'tailChord'] = b_t * rho_t * g
        partials['tailWeight', 'tailMassPerArea'] = c_t * b_t * g


class BoomMassAndWeight(om.ExplicitComponent):

    def setup(self):

        self.add_input('tailDistance', units='m')
        self.add_input('boomMassPerLength', units='kg/m')
        self.add_input('g', units='m/s**2')

        self.add_output('boomMass', units='kg')
        self.add_output('boomWeight', units='N')

        self.declare_partials(['boomMass'], ['tailDistance'])
        self.declare_partials(['boomWeight'], ['tailDistance'])

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        l_t = inputs['tailDistance']
        rho_boom = inputs['boomMassPerLength']
        g = inputs['g']

        m_boom = l_t * rho_boom
        W_boom = m_boom * g

        outputs['boomMass'] = m_boom
        outputs['boomWeight'] = W_boom

    def compute_partials(self, inputs, partials, discrete_inputs=None):

        rho_boom = inputs['boomMassPerLength']
        g = inputs['g']

        partials['boomMass', 'tailDistance'] = rho_boom

        partials['boomWeight', 'tailDistance'] = rho_boom * g


class TotalCGDistance(om.ExplicitComponent):

    def setup(self):

        self.add_input('tailChord', units='m')
        self.add_input('restOfAircraftCGDistance', units='m')
        self.add_input('restOfAircraftMass', units='kg')
        self.add_input('wingMass', units='kg')
        self.add_input('tailMass', units='kg')
        self.add_input('boomMass', units='kg')
        self.add_input('tailDistance', units='m')
        self.add_input('wingDistance', units='m')
        self.add_input('wingACDistance', units='m')

        self.add_output('totalCGDistance', units='m')

        self.declare_partials(['totalCGDistance'], ['wingMass', 'tailMass', 'boomMass', 'tailDistance', 'wingDistance', 'tailChord'], method='cs')

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        c_t = inputs['tailChord']
        CG_ac = inputs['restOfAircraftCGDistance']
        x_t = inputs['tailDistance']
        x_w = inputs['wingDistance']
        ho_c = inputs['wingACDistance']
        m_ac = inputs['restOfAircraftMass']
        m_w = inputs['wingMass']
        m_t = inputs['tailMass']
        m_b = inputs['boomMass']

        CG_b = x_t*0.5
        CG_t = x_t+(0.25*c_t)
        CG_w = (x_w + 0.408)

        CG_tot = (m_ac*CG_ac + m_w*CG_w + m_t*CG_t + m_b*CG_b) / (m_ac + m_w + m_t + m_b)

        outputs['totalCGDistance'] = CG_tot


class WeightsGroup(om.Group):

    def setup(self):

        self.add_subsystem('tailMassAndWeight_sys', TailMassAndWeight(),
                           promotes=['*'])
        self.add_subsystem('boomMassAndWeight_sys', BoomMassAndWeight(),
                           promotes=['*'])
        self.add_subsystem('totalCGDistance_sys', TotalCGDistance(),
                           promotes=['*'])