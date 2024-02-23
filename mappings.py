STATE_SLUG_MAPPING: dict[str, str] = {
    "d-alaska": "Alaska",
    "d-alaska-1": "Alaska",
    "d-ala": "Alabama",
    "md-ala": "Alabama",
    "nd-ala": "Alabama",
    "sd-ala": "Alabama",
    "sd-ala-1": "Alabama",
    "d-ariz": "Arizona",
    "d-ark": "Arkansas",
    "ed-ark": "Arkansas",
    "wd-ark": "Arkansas",
    "wd-ark-1": "Arkansas",
    "d-cal": "California",
    "cd-cal": "California",
    "ed-cal": "California",
    "nd-cal": "California",
    "sd-cal": "California",
    "ccsd-cal": "California",
    "d-colo": "Colorado",
    "d-conn": "Connecticut",
    "d-del": "Delaware",
    "dcc": "D.C.",
    "ddc": "D.C.",
    "ddc-2": "D.C.",
    "ddc-3": "D.C.",
    "ddc-4": "D.C.",
    "dc-6": "D.C.",
    "d-fla": "Florida",
    "md-fla": "Florida",
    "nd-fla": "Florida",
    "sd-fla": "Florida",
    "d-ga": "Georgia",
    "md-ga": "Georgia",
    "nd-ga": "Georgia",
    "sd-ga": "Georgia",
    "d-haw": "Hawaii",
    "d-idaho": "Idaho",
    "cd-ill": "Illinois",
    "ed-ill": "Illinois",
    "nd-ill": "Illinois",
    "nd-ill-1": "Illinois",
    "sd-ill": "Illinois",
    "nd-ind": "Indiana",
    "sd-ind": "Indiana",
    "nd-iowa": "Iowa",
    "sd-iowa": "Iowa",
    "d-kan": "Kansas",
    "ed-kan-1": "Kansas",
    "ed-ky": "Kentucky",
    "wd-ky": "Kentucky",
    "wd-ky-1": "Kentucky",
    "d-la": "Louisiana",
    "ed-la": "Louisiana",
    "md-la": "Louisiana",
    "wd-la": "Louisiana",
    "d-me": "Maine",
    "d-md": "Maryland",
    "d-mass-1": "Massachusetts",
    "d-mass-2": "Massachusetts",
    "d-mich": "Michigan",
    "ed-mich": "Michigan",
    "wd-mich": "Michigan",
    "d-minn": "Minnesota",
    "d-minn-1": "Minnesota",
    "nd-miss": "Mississippi",
    "sd-miss": "Mississippi",
    "d-mo": "Missouri",
    "ed-mo": "Missouri",
    "wd-mo": "Missouri",
    "wd-mo-1": "Missouri",
    "sd-mo": "Missouri",
    "d-mont": "Montana",
    "d-neb": "Nebraska",
    "d-nev": "Nevada",
    "dnh": "New Hampshire",
    "dnj": "New Jersey",
    "d-nj": "New Jersey",
    "dnm": "New Mexico",
    "dnm-1": "New Mexico",
    "dny": "New York",
    "edny": "New York",
    "edny-1": "New York",
    "ndny": "New York",
    "wdny": "New York",
    "sdny": "New York",
    "sdny-1": "New York",
    "dnc": "North Carolina",
    "ednc": "North Carolina",
    "mdnc": "North Carolina",
    "wdnc": "North Carolina",
    "dnd": "North Dakota",
    "d-nd": "North Dakota",
    "nd-ohio": "Ohio",
    "sd-ohio": "Ohio",
    "wd-ohio": "Ohio",
    "ed-okla": "Oklahoma",
    "nd-okla": "Oklahoma",
    "wd-okla": "Oklahoma",
    "d-or": "Oregon",
    "d-pa": "Pennsylvania",
    "ed-pa": "Pennsylvania",
    "ed-pa-1": "Pennsylvania",
    "md-pa": "Pennsylvania",
    "wd-pa": "Pennsylvania",
    "dpr": "Puerto Rico",
    "d-pr": "Puerto Rico",
    "d-pr-1": "Puerto Rico",
    "dpr-1": "Puerto Rico",
    "dri": "Rhode Island",
    "edsc": "South Carolina",
    "dsc": "South Carolina",
    "wdsc": "South Carolina",
    "dsd": "South Dakota",
    "d-sd": "South Dakota",
    "d-sd-1": "South Dakota",
    "d-sd-2": "South Dakota",
    "d-sd-3": "South Dakota",
    "d-sd-4": "South Dakota",
    "d-tenn": "Tennessee",
    "ed-tenn": "Tennessee",
    "md-tenn": "Tennessee",
    "wd-tenn": "Tennessee",
    "d-tex": "Texas",
    "ed-tex": "Texas",
    "nd-tex": "Texas",
    "sd-tex": "Texas",
    "wd-tex": "Texas",
    "d-utah": "Utah",
    "d-vt": "Vermont",
    "dvi": "Virgin Islands",
    "vi-dist-ct": "Virgin Islands",
    "d-va": "Virginia",
    "ed-va": "Virginia",
    "ndw-va": "Virginia",
    "sdw-va-1": "Virginia",
    "d-wash": "Washington",
    "ed-wash": "Washington",
    "wd-wash": "Washington",
    "dw-va": "West Virginia",
    "wd-va": "West Virginia",
    "sd-va": "West Virginia",
    "sd-wva": "West Virginia",
    "sd-wva-1": "West Virginia",
    "sd-wva-2": "West Virginia",
    "sd-wva-3": "West Virginia",
    "sd-wva-4": "West Virginia",
    "nd-wva": "West Virginia",
    "sdw-va": "West Virginia",
    "d-wis": "Wisconsin",
    "ed-wis": "Wisconsin",
    "wd-wis": "Wisconsin",
    "d-wyo": "Wyoming",
    "d-wyo-1": "Wyoming",
    "d-guam": "misc",  # Guam (too small)
    "d-guam-1": "misc",  # Guam (too small)
    "d-n-mar-i": "misc",  # North Mariana Islands (too small)
    "dcz": "misc",  # United States District Court for the District of the Canal Zone
    "ct-cl": "misc",  # Court of Claims
    "cl-ct": "misc",  # Court of Claims
    "ct-intl-trade": "misc",  # International Trade
    "cust-ct": "misc",  # Customs Court
    "ct-cust": "misc",  # Customs Court
    "jpml-2": "misc",  # Multidistrict Litigation
    "uscmr": "misc",  # United States Court of Military Commission Review
    "regl-rail-reorg-ct": "misc",  # Special Court, Regional Rail Reorganization Act
    "regl-rail-reorg-ct-1": "misc",  # Special Court, Regional Rail Reorganization Act
    "dc-3": "misc",  # District of Columbia Municipal Court of Appeals (?)
    "dc-5": "misc",  # DC Circuit (not sure why included in F. Supp.)
    "dc-cir-4": "misc",  # DC Circuit (not sure why included in F. Supp.)
    "dc-cir-6": "misc",  # DC Circuit (not sure why included in F. Supp.)
    "dc-cir-8": "misc",  # DC Circuit (not sure why included in F. Supp.)
    "usdc-2": "misc",  # DC Circuit (not sure why included in F. Supp.)
    "ccwdny": "misc",  # PA Circuit (?) (not sure why included in F. Supp.)
    "ccd-del": "misc",  # Delaware Circuit (?) (not sure why included in F. Supp.)
    "cced-mo": "misc",  # Missouri Circuit (?) (not sure why included in F. Supp.)
    "ccwd-mo": "misc",  # Missouri Circuit (?) (not sure why included in F. Supp.)
    "ccsd-miss": "misc",  # Mississippi Circuit (?) (not sure why included in F. Supp.)
    "cced-ark": "misc",  # Arkansas Circuit (?) (not sure why included in F. Supp.)
    "9th-cir-1": "misc",  # 9th Circuit (not sure why included in F. Supp.)
    "bankr-ed-pa": "misc",  # Bankrupcy (not sure why included in F. Supp.)
    "bankr-wd-pa": "misc",  # Bankrupcy (not sure why included in F. Supp.)
    "bankr-ed-mich": "misc",  # Bankrupcy (not sure why included in F. Supp.)
    "bankr-d-or": "misc",  # Bankrupcy (not sure why included in F. Supp.)
    "bankr-sd-tex": "misc",  # Bankrupcy (not sure why included in F. Supp.)
    "bankr-d-kan": "misc",  # Bankrupcy (not sure why included in F. Supp.)
    "bankr-wd-wash": "misc",  # Bankrupcy (not sure why included in F. Supp.)
    "bankr-d-haw": "misc",  # Bankrupcy (not sure why included in F. Supp.)
    "bankr-sd-cal": "misc",  # Bankrupcy (not sure why included in F. Supp.)
    "bankr-d-mont": "misc",  # Bankrupcy (not sure why included in F. Supp.)
    "bankr-cd-cal": "misc",  # Bankrupcy (not sure why included in F. Supp.)
    "bankr-ed-mo": "misc",  # Bankrupcy (not sure why included in F. Supp.)
    "misc": "misc",
}

CAP_CIRCUIT_MAPPING: dict[int, int] = {
    8809: 1,
    10715: 1,
    8778: 2,
    9993: 2,
    8840: 3,
    9319: 3,
    11846: 3,
    8954: 4,
    8820: 5,
    8864: 6,
    21296: 6,
    9010: 7,
    8821: 8,
    8826: 9,
    18869: 9,
    8771: 10,
    9031: 11,
    8770: 12,  # DC
    9000: 12,  # DC
    9509: 12,  # DC
    10893: 12,  # DC
    11429: 12,  # DC
    11448: 12,  # DC
    11500: 12,  # DC
    17013: 12,  # DC
    8955: 13,  # Fed
    17247: 99,  # FISA
    22881: 99,  # FISA
    9000: 99,  # DC District Court (included in CAP data for some reason)
    15799: 99,  # United States Judicial Conference Committee on Judicial Conduct and Disability
    13783: 99,  # United States Judicial Conference Committee to Review Circuit Council Conduct and Disability Orders
    21005: 99,  # 2nd cir Judicial council
    15811: 99,  # 3rd cir Judicial council
    15077: 99,  # 6th cir Judicial council
    15332: 99,  # 7th Judicial council
    20154: 99,  # 8th cir Judicial council
    12042: 99,  # 9th cir Judicial council
    22329: 99,  # DC cir Judicial council
    15102: 99,  # DC cir judicial council
}
