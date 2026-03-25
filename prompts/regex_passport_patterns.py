LABEL_PATTERNS = {

    # --- NAME FIELDS ---
    "SURNAME": [
        r"SURNAME", r"MBIEMRI", r"UZV[ĀA]RDS", r"PRIEZVISKO", r"UZVARDS"
    ],
    "GIVEN NAME": [
        r"GIVEN\s*NAMES", r"EMRI", r"V[ĀA]RDS", r"MENO", r"GIVEN\s*NAMES?"
    ],

    # --- NATIONALITY ---
    "NATIONALITY": [
        r"NATIONALITY", r"SHTETESIA", r"PILSONIBA", r"ŠT[ÁA]TNE\s*OBČIANSTVO", r"OBČIANSTVO", r"VALSTS"
    ],

    # --- DATES ---
    "DATE OF BIRTH": [
        r"DATE\s*OF\s*BIRTH", r"BIRTH", r"LINDJA", r"DZIM", r"NAROD", r"DZIMŠANAS\s*DATUMS"
    ],
    "DATE OF ISSUE": [
        r"DATE\s*OF\s*ISSUE", r"ISSUE", r"LESHIMIT", r"IZDOŠ", r"VYDANIA", r"IZSniegšanas"
    ],
    "DATE OF EXPIRY": [
        r"DATE\s*OF\s*EXPIRY", r"EXPIRY", r"SKADIMIT", r"DER[ĪI]GUMA", r"PLATNOSTI", r"DERĪGS\s*LĪDZ"
    ],

    # --- SEX ---
    "SEX": [
        r"SEX", r"GJINIA?", r"DZIMUMS", r"POHLAVIE"
    ],

    # --- DOCUMENT NUMBERS ---
    "ID CARD NO": [
        r"ID\s*CARD\s*NO", r"CARD\s*NO", r"NR\.\s*LET", r"Č[ÍI]SLO", r"\bNO\.\b"
    ],
    "PASSPORT NO": [
        r"PASSPORT\s*NO", r"PASE", r"PASSPORT", r"NR", r"NUM", r"PAS\.?\s*NR"
    ],
    "PERSONAL NO": [
        r"PERSONAL\s*NO", r"KODS", r"PERSONAS\s*KODS", r"RODN[ÉE]\s*Č[ÍI]SLO"
    ],

    # --- LOCATION ---
    "PLACE OF BIRTH": [
        r"PLACE\s*OF\s*BIRTH", r"BIRTHPLACE", r"VENDLINDJA",
        r"DZIMŠANAS\s*VIETA", r"MIESTO\s*NARODENIA", r"LIEU\s*DE\s*NAISSANCE"
    ],

    # --- AUTHORITY / ISSUER ---
    "AUTHORITY": [
        r"AUTHORITY", r"AUTORITETI", r"AUTOR", r"IEST[ĀA]DE",
        r"IZDEV[ĒE]JS", r"ISSUED\s*BY", r"VYD[ÁA]L", r"IZSNIEDZA"
    ],

    # --- HEIGHT ---
    "HEIGHT": [
        r"HEIGHT", r"TAILLE", r"AUGUMS", r"\bCM\b"
    ],

    # Passport Type (Latvia)
    "TYPE": [
        r"\bTYPE\b", r"TIPS?", r"TIPS"
    ],

    # Issuing State Code (Latvia)
    "ISSUING STATE CODE": [
        r"CODE\s*OF\s*ISSUING\s*STATE",
        r"VALSTS\s*KODS",
        r"CODE\s*DU\s*PAYS"
    ],

    # Issued By (Slovakia specific)
    "ISSUED BY": [
        r"ISSUED\s*BY", r"VYDAL", r"IZDEVIS", r"ISSUER"
    ],

    # Signature presence (no OCR value, but label detection)
    "SIGNATURE": [
        r"SIGNATURE", r"FIRMA", r"PARAKSTS", r"PODPIS", r"HOLDER'?S\s*SIGNATURE"
    ]
}