

symbolset = ["/", "-", "\\", "|", "/", "-", "\\", "|"]
nsymbols = len(symbolset)
def print_progress(iter, imax, nbars = 20, start="", end = ""):
    ibar = int((iter/imax)*nbars)
    isymbol = int((ibar + 1 - (iter/imax)*nbars)*nsymbols)%nsymbols
    string = start + "["+"*"*ibar + symbolset[isymbol]+" "*(nbars-(ibar+1))+"]" + end
    print(string, end="\r")
    if iter == imax - 1:
        print("\n")