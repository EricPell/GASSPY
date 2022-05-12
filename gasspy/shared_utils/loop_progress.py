

symbolset = ["/", "-", "\\", "|", "/", "-", "\\", "|"]
nsymbols = len(symbolset)
def print_progress(iter, imax, nbars = 20, start=""):
    ibar = int((iter/imax)*nbars)
    isymbol = int((ibar + 1 - (iter/imax)*nbars)*nsymbols)%nsymbols
    string = start + "["+"*"*ibar + symbolset[isymbol]+" "*(nbars-(ibar+1))+"]"
    print(string, end="\r")
    if iter == imax - 1:
        print("\n")