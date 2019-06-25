

chromoson = {"Winner": {}}

gen =  {"a":1, "b" : "ja", "c" : 3}
gen1 =  {"a": 2, "b" : "ja", "c" : 4}


for i in range(0,2):
    chromoson["Winner"][i] = gen
    chromoson["Winner"][i] = gen1

print(chromoson)