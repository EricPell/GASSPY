""" gasspy DECISION TREE """
import astropy.units as u
class tree(object):
    def __init(self):
        self.T_collisional_ionization_limit = 16e3*u.K

        # This initializes the results of the test 
        self.physics = {"shocked":False,
        "fully_ionized":False,
        "ionization_front":False,
        "shielded":False,m  
        "turbulent":False}

        # This creates a dictionary which contains links to the test.
        # Each test will return a True or False per cell.
        self.tests = {"shocked":self.shocked,
        "fully_ionized":self.fully_ionized,
        "ionization_front":self.ionization_front,
        "shielded":self.shielded,
        "turbulent":self.turbulent}

    def add_physics(self, name, function):
        pass

    def find_branch(self, nH, T, **kwargs):
        # This is a loop that prevents the physics checks from being hardcoded.
        # After initialization of the tree object a user can add a new function by linking it to tests.
        for process in self.physics.keys():
            self.tests[process]()

    def shocked(self):
        if self.T > self.T_collisional_ionization_limit:
            self.physics["shocked"] = True

    def turbulent(self):
        pass

    def Prad_grad(self, depth, phi0, n, EeV=13.6, T4=0.8, beta=2, gama=10):
        n0 = 4.5 * 10**5 * (T4**4.66/Q049) * (18/EeV)**3

        P0 = n0*10**4
        dP_dl = self.alpha * n0**2
        dn_dl = dP_dl / T
        
    