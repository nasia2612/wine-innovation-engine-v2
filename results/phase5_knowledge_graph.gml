graph [
  directed 1
  node [
    id 0
    label "body_score"
    type "outcome"
    size 10
  ]
  node [
    id 1
    label "organic_acid"
    type "family"
    size 10
  ]
  node [
    id 2
    label "fatty_acid"
    type "family"
    size 10
  ]
  node [
    id 3
    label "amino_acid_derivative"
    type "family"
    size 10
  ]
  node [
    id 4
    label "fatty_alcohol"
    type "family"
    size 10
  ]
  node [
    id 5
    label "sugar_polyol"
    type "family"
    size 10
  ]
  node [
    id 6
    label "nucleobase"
    type "family"
    size 10
  ]
  node [
    id 7
    label "fermentation_progress"
    type "process"
    size 10
  ]
  node [
    id 8
    label "yeast_autolysis"
    type "process"
    size 10
  ]
  node [
    id 9
    label "grape_ripeness"
    type "process"
    size 10
  ]
  node [
    id 10
    label "2-ketoisocaproic acid"
    type "metabolite"
    vip 2.008796344447942
    direction "lower body"
    size 10
  ]
  node [
    id 11
    label "palmitic acid"
    type "metabolite"
    vip 1.9377523718012213
    direction "lower body"
    size 10
  ]
  node [
    id 12
    label "1-hexadecanol"
    type "metabolite"
    vip 1.8631987031128885
    direction "lower body"
    size 10
  ]
  node [
    id 13
    label "stearic acid"
    type "metabolite"
    vip 1.8484202080443168
    direction "lower body"
    size 10
  ]
  node [
    id 14
    label "octadecanol"
    type "metabolite"
    vip 1.83745627335311
    direction "lower body"
    size 10
  ]
  node [
    id 15
    label "tartaric acid"
    type "metabolite"
    vip 1.7449823364985402
    direction "lower body"
    size 10
  ]
  node [
    id 16
    label "arachidic acid"
    type "metabolite"
    vip 1.713181462722906
    direction "lower body"
    size 10
  ]
  node [
    id 17
    label "ribose"
    type "metabolite"
    vip 1.6777785815917632
    direction "lower body"
    size 10
  ]
  node [
    id 18
    label "inositol myo-"
    type "metabolite"
    vip 1.617276519298489
    direction "lower body"
    size 10
  ]
  node [
    id 19
    label "4-hydroxyproline"
    type "metabolite"
    vip 1.6102799692572414
    direction "higher body"
    size 10
  ]
  node [
    id 20
    label "3-hydroxy-3-methylglutaric acid"
    type "metabolite"
    vip 1.589415012546908
    direction "lower body"
    size 10
  ]
  node [
    id 21
    label "sophorose"
    type "metabolite"
    vip 1.5730518423567603
    direction "lower body"
    size 10
  ]
  node [
    id 22
    label "isocitric acid"
    type "metabolite"
    vip 1.5628523539075465
    direction "lower body"
    size 10
  ]
  node [
    id 23
    label "glycerol"
    type "metabolite"
    vip 1.552782911037381
    direction "lower body"
    size 10
  ]
  node [
    id 24
    label "heptadecanoic acid"
    type "metabolite"
    vip 1.552245699368686
    direction "lower body"
    size 10
  ]
  node [
    id 25
    label "xylitol"
    type "metabolite"
    vip 1.5394353090516195
    direction "lower body"
    size 10
  ]
  node [
    id 26
    label "uracil"
    type "metabolite"
    vip 1.504952333106622
    direction "lower body"
    size 10
  ]
  node [
    id 27
    label "fucose"
    type "metabolite"
    vip 1.4780207883606455
    direction "lower body"
    size 10
  ]
  node [
    id 28
    label "maleic acid"
    type "metabolite"
    vip 1.4476987025564911
    direction "higher body"
    size 10
  ]
  node [
    id 29
    label "glyceric acid"
    type "metabolite"
    vip 1.4277921822770434
    direction "lower body"
    size 10
  ]
  edge [
    source 7
    target 2
    relation "consumes"
    note "Palmitic/stearic decline through fermentation (Skogerson 2009, refs 31-32)"
    width 1
  ]
  edge [
    source 7
    target 4
    relation "consumes"
    note "Palmitic/stearic decline through fermentation (Skogerson 2009, refs 31-32)"
    width 1
  ]
  edge [
    source 8
    target 5
    relation "produces"
    note "Ribose, myo-inositol appear as yeast dies (Phase 1 Xt state)"
    width 1
  ]
  edge [
    source 9
    target 3
    relation "produces"
    note "Structural amino acids from grape phenolic maturity"
    width 1
  ]
  edge [
    source 10
    target 0
    relation "decreases"
    width 2.008796344447942
  ]
  edge [
    source 10
    target 1
    relation "belong_to"
    width 1
  ]
  edge [
    source 10
    target 11
    relation "correlates_with"
    width 0.872521796636553
  ]
  edge [
    source 10
    target 13
    relation "correlates_with"
    width 0.8633854435163695
  ]
  edge [
    source 10
    target 15
    relation "correlates_with"
    width 0.8636617893260047
  ]
  edge [
    source 11
    target 0
    relation "decreases"
    width 1.9377523718012213
  ]
  edge [
    source 11
    target 2
    relation "belong_to"
    width 1
  ]
  edge [
    source 11
    target 13
    relation "correlates_with"
    width 0.9708626434386952
  ]
  edge [
    source 11
    target 15
    relation "correlates_with"
    width 0.9217441303732694
  ]
  edge [
    source 11
    target 16
    relation "correlates_with"
    width 0.9170121366486663
  ]
  edge [
    source 11
    target 24
    relation "correlates_with"
    width 0.9123455415688602
  ]
  edge [
    source 12
    target 0
    relation "decreases"
    width 1.8631987031128885
  ]
  edge [
    source 12
    target 4
    relation "belong_to"
    width 1
  ]
  edge [
    source 13
    target 0
    relation "decreases"
    width 1.8484202080443168
  ]
  edge [
    source 13
    target 2
    relation "belong_to"
    width 1
  ]
  edge [
    source 13
    target 15
    relation "correlates_with"
    width 0.8916976919897551
  ]
  edge [
    source 13
    target 16
    relation "correlates_with"
    width 0.8544946641382095
  ]
  edge [
    source 13
    target 24
    relation "correlates_with"
    width 0.8571649972330551
  ]
  edge [
    source 14
    target 0
    relation "decreases"
    width 1.83745627335311
  ]
  edge [
    source 14
    target 4
    relation "belong_to"
    width 1
  ]
  edge [
    source 15
    target 0
    relation "decreases"
    width 1.7449823364985402
  ]
  edge [
    source 15
    target 1
    relation "belong_to"
    width 1
  ]
  edge [
    source 15
    target 16
    relation "correlates_with"
    width 0.9041664147463508
  ]
  edge [
    source 15
    target 24
    relation "correlates_with"
    width 0.8638997648577645
  ]
  edge [
    source 16
    target 0
    relation "decreases"
    width 1.713181462722906
  ]
  edge [
    source 16
    target 2
    relation "belong_to"
    width 1
  ]
  edge [
    source 16
    target 24
    relation "correlates_with"
    width 0.9514036459584835
  ]
  edge [
    source 17
    target 0
    relation "decreases"
    width 1.6777785815917632
  ]
  edge [
    source 17
    target 5
    relation "belong_to"
    width 1
  ]
  edge [
    source 18
    target 0
    relation "decreases"
    width 1.617276519298489
  ]
  edge [
    source 18
    target 5
    relation "belong_to"
    width 1
  ]
  edge [
    source 19
    target 0
    relation "increases"
    width 1.6102799692572414
  ]
  edge [
    source 19
    target 3
    relation "belong_to"
    width 1
  ]
  edge [
    source 20
    target 0
    relation "decreases"
    width 1.589415012546908
  ]
  edge [
    source 20
    target 1
    relation "belong_to"
    width 1
  ]
  edge [
    source 21
    target 0
    relation "decreases"
    width 1.5730518423567603
  ]
  edge [
    source 21
    target 5
    relation "belong_to"
    width 1
  ]
  edge [
    source 22
    target 0
    relation "decreases"
    width 1.5628523539075465
  ]
  edge [
    source 22
    target 1
    relation "belong_to"
    width 1
  ]
  edge [
    source 23
    target 0
    relation "decreases"
    width 1.552782911037381
  ]
  edge [
    source 24
    target 0
    relation "decreases"
    width 1.552245699368686
  ]
  edge [
    source 24
    target 2
    relation "belong_to"
    width 1
  ]
  edge [
    source 25
    target 0
    relation "decreases"
    width 1.5394353090516195
  ]
  edge [
    source 25
    target 5
    relation "belong_to"
    width 1
  ]
  edge [
    source 26
    target 0
    relation "decreases"
    width 1.504952333106622
  ]
  edge [
    source 26
    target 6
    relation "belong_to"
    width 1
  ]
  edge [
    source 27
    target 0
    relation "decreases"
    width 1.4780207883606455
  ]
  edge [
    source 27
    target 5
    relation "belong_to"
    width 1
  ]
  edge [
    source 28
    target 0
    relation "increases"
    width 1.4476987025564911
  ]
  edge [
    source 28
    target 1
    relation "belong_to"
    width 1
  ]
  edge [
    source 29
    target 0
    relation "decreases"
    width 1.4277921822770434
  ]
  edge [
    source 29
    target 5
    relation "belong_to"
    width 1
  ]
]
