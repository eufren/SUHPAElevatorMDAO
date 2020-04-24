import openmdao.api as om


class TotalVerticalForce(om.ExplicitComponent):

    def setup(self):

        self.add_input('wingLift', units='N')
        self.add_input('tailLift', units='N')
        self.add_input('tailWeight', units='N')
        self.add_input('boomWeight', units='N')
        self.add_input('restOfAircraftMass', units='kg')
        self.add_input('wingMass', units='kg')
        self.add_input('g', units='m/s**2')

        self.add_output('verticalForce', units='N')

        self.declare_partials(['verticalForce'], ['wingLift', 'tailLift', 'tailWeight', 'boomWeight'])

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        L = inputs['wingLift']
        L_t = inputs['tailLift']
        W_t = inputs['tailWeight']
        W_b = inputs['boomWeight']
        m_w = inputs['wingMass']
        m_ac = inputs['restOfAircraftMass']
        g = inputs['g']

        W_w = m_w * g
        W_ac = m_ac * g

        W_tot = W_ac + W_w + W_t + W_b

        L_tot = L + L_t

        F_tot = L_tot - W_tot

        outputs['verticalForce'] = F_tot

    def compute_partials(self, inputs, partials, discrete_inputs=None):

        partials['verticalForce', 'wingLift'] = 1
        partials['verticalForce', 'tailLift'] = 1
        partials['verticalForce', 'tailWeight'] = -1
        partials['verticalForce', 'boomWeight'] = -1


class TotalDrag(om.ExplicitComponent):

    def setup(self):

        self.add_input('wingDrag', units='N')
        self.add_input('tailInducedDrag', units='N')
        self.add_input('tailSkinDrag', units='N')

        self.add_output('totalDrag', units='N')

        self.declare_partials(['totalDrag'], ['wingDrag', 'tailInducedDrag', 'tailSkinDrag'])

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        D_w = inputs['wingDrag']
        Di_t = inputs['tailInducedDrag']
        Df_t = inputs['tailSkinDrag']

        D_tot = D_w + Di_t + Df_t

        outputs['totalDrag'] = D_tot

    def compute_partials(self, inputs, partials, discrete_inputs=None):

        partials['totalDrag', 'wingDrag'] = 1
        partials['totalDrag', 'tailInducedDrag'] = 1
        partials['totalDrag', 'tailSkinDrag'] = 1


class TotalMomentAboutCG(om.ExplicitComponent):

    def setup(self):

        self.add_input('totalCGDistance', units='m')
        self.add_input('wingDistance', units='m')
        self.add_input('tailDistance', units='m')
        self.add_input('wingLift', units='N')
        self.add_input('tailLift', units='N')
        self.add_input('wingMoment', units='N*m')
        self.add_input('wingACDistance', units='m')
        self.add_input('tailChord', units='m')

        self.add_output('momentAboutCG', units='N*m')

        self.declare_partials(['momentAboutCG'], ['*'])

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        M_0 = inputs['wingMoment']
        x_CG = inputs['totalCGDistance']
        x_w = inputs['wingDistance']
        ho_c = inputs['wingACDistance']
        x_t = inputs['tailDistance']
        L = inputs['wingLift']
        L_t = inputs['tailLift']
        c_t = inputs['tailChord']

        M_CG = M_0 + (x_CG-(x_w+ho_c))*L - ((x_t+0.25*c_t)-x_CG)*L_t

        outputs['momentAboutCG'] = M_CG

    def compute_partials(self, inputs, partials, discrete_inputs=None):

        x_CG = inputs['totalCGDistance']
        x_w = inputs['wingDistance']
        ho_c = inputs['wingACDistance']
        x_t = inputs['tailDistance']
        c_t = inputs['tailChord']
        L = inputs['wingLift']
        L_t = inputs['tailLift']

        partials['momentAboutCG', 'wingMoment'] = 1
        partials['momentAboutCG', 'tailChord'] = -0.25*L_t
        partials['momentAboutCG', 'wingLift'] = x_CG - x_w - ho_c
        partials['momentAboutCG', 'tailLift'] = -(x_t+0.25*c_t)+x_CG
        partials['momentAboutCG', 'totalCGDistance'] = L + L_t
        partials['momentAboutCG', 'wingDistance'] = - L
        partials['momentAboutCG', 'wingACDistance'] = - L
        partials['momentAboutCG', 'tailDistance'] = - L_t


class TailVolume(om.ExplicitComponent):

    def setup(self):

        self.add_input('tailChord')
        self.add_input('tailSpan')
        self.add_input('tailDistance')
        self.add_input('wingArea')
        self.add_input('wingMAC')

        self.add_output('tailVolume')

        self.declare_partials(['*'], ['*'], method="cs")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        c_t = inputs['tailChord']
        b_t = inputs['tailSpan']
        l_t = inputs['tailDistance']
        S = inputs['wingArea']
        c = inputs['wingMAC']

        S_t = b_t*c_t
        C_H = (S_t*l_t)/(S*c)

        outputs['tailVolume'] = C_H


class StaticMargin(om.ExplicitComponent):

    def setup(self):

        self.add_input('tailSpan', units='m')
        self.add_input('tailChord', units='m')
        self.add_input('tailVolume')
        self.add_input('wingMAC', units='m')
        self.add_input('wingACDistance', units='m')
        self.add_input('dCL_dAlpha', units='rad**-1')
        self.add_input('taildCLdAlpha', units='rad**-1')
        self.add_input('wingArea', units='m**2')
        self.add_input('totalCGDistance', units='m')
        self.add_input('wingDistance', units='m')

        self.add_output('staticMargin')

        self.declare_partials(['*'], ['*'], method='cs')

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        b_t = inputs['tailSpan']
        c_t = inputs['tailChord']
        x_CG = inputs['totalCGDistance']
        x_w = inputs['wingDistance']
        c = inputs['wingMAC']
        ho_c = inputs['wingACDistance']
        CL_a = inputs['dCL_dAlpha']
        CL_at = inputs['taildCLdAlpha']
        S = inputs['wingArea']
        K = inputs['tailVolume']

        S_t = b_t*c_t
        CL_aStar = CL_a + CL_at*(S_t/S)
        h_c = x_CG - x_w
        h = h_c / c
        ho = ho_c / c

        H_s = ho-h+K*(CL_at/CL_aStar)

        outputs['staticMargin'] = H_s