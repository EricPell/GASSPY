import astropy
from astropy.modeling import models
from astropy import units as u
from math import floor, log10
import numpy as np

class gasspy_bb():
    def __init__(self, E_unit=u.eV, padding=1e-3, T=1e5, N_samples=10):
        if isinstance(E_unit, str):
            E_unit = u.Unit(E_unit)
        self.E_unit=E_unit
        self.padding=1e-5
        self.T = T
        self.N_samples = N_samples
        self.bb = models.BlackBody(temperature=self.T*u.K)
        self.field_name = None

        assert self.padding < 1.0, "what did you do?"

    def make_SED(self, Emin, Emax, N_samples=None, T=None, writetosed=True, outname=None, field_name=None):
        if T != None:
            self.T = T
            self.bb = models.BlackBody(temperature=self.T*u.K)

        if not isinstance(Emin, astropy.units.quantity.Quantity):
            Emin = Emin * self.E_unit 
        else:
            # Check for dimensionless
            if Emin.unit == u.Unit():
                Emin = Emin * self.E_unit
            elif Emin.unit != self.E_unit:
                Emin = Emin.to(u.eV, equivalencies=u.spectral())

        if not isinstance(Emax, astropy.units.quantity.Quantity):
            Emax = Emax * self.E_unit 
        else:
            # Check for dimensionless
            if Emax.unit == u.Unit():
                Emax = Emax * self.E_unit
            elif Emax.unit != self.E_unit:
                Emax = Emax.to(u.eV, equivalencies=u.spectral())

        # Emin and max could be reversed do to a switch from wavelength to frequency/energy
        Emin, Emax = min(Emin, Emax), max(Emin, Emax)

        if self.N_samples > 1:
            E_array = np.linspace(Emin, Emax, self.N_samples)
            Flux_array = self.bb(E_array)
        else:
            Emid = (Emax - Emin) / 2.0
            E_array = np.array([Emin.value, Emax.value])*self.E_unit
            Flux_array = self.bb(np.array([Emid, Emid])*self.E_unit)

        if writetosed:
            self.write_SED(E_array, Flux_array, outname=outname, field_name=field_name)

    def cloudy_SED_list(self, E_array, Flux_array):
        E_array = E_array.value
        Flux_array = Flux_array.value
        dE = (E_array[-1] - E_array[0]) * self.padding
        out_lines = ["%f %e fnu unit eV \n"%(E_array[0] - dE, 1e-35.0),]
        out_lines += ["%f %e\n"%(E_array[i], Flux_array[i]) for i in range(len(E_array))]
        out_lines +=  ["%f %e\n"%(E_array[-1] + dE, 1e-35.0),]
        return(out_lines)

    def write_SED(self, E_array, Flux_array, outname=None, field_name=None):
        if outname == None:
            prec = int(-floor(log10(self.padding)))

            if field_name == None:
                if self.field_name != None:
                    field_name = self.field_name
                else:
                    field_name = "_Emin_%f_Emax_%f"%(E_array[0].value, E_array[-1].value)
            else:
                field_name = "_"+field_name

            outname = "gasspy_blackbody_T_"+f"{self.T:.{prec}f}"+ field_name + ".sed"

        f_out = open(outname, "w")

        f_out.writelines(self.cloudy_SED_list(E_array, Flux_array))
        f_out.writelines("\n")
        f_out.close()
        
        return(outname)

if __name__ == "__main__":
    T = 1e6
    E1 = 0.1 * u.eV
    E2 = 10e3 * u.eV
    mybb = gasspy_bb(T=T)
    mybb.make_SED(E1, E2, writetosed=True, field_name="hardXrays")