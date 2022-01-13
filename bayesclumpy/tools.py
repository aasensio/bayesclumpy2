def _lower_to_sep(string, separator='='):
    line = string.partition(separator)
    string = str(line[0]).lower()+str(line[1])+str(line[2])
    return string

def tobool(l):
    return True if l == 'True' else False