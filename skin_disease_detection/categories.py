# Universal Category Mapping for all 23 DermNet Folders
CATEGORIES = [
    'Acne and Rosacea Photos',
    'Actinic Keratosis Basal Cell Carcinoma and other Malignant Lesions',
    'Atopic Dermatitis Photos',
    'Bullous Disease Photos',
    'Cellulitis Impetigo and other Bacterial Infections',
    'Eczema Photos',
    'Exanthems and Drug Eruptions',
    'Hair Loss Photos Alopecia and other Hair Diseases',
    'Herpes HPV and other STDs Photos',
    'Light Diseases and Disorders of Pigmentation',
    'Lupus and other Connective Tissue diseases',
    'Melanoma Skin Cancer Nevi and Moles',
    'Nail Fungus and other Nail Disease',
    'Poison Ivy Photos and other Contact Dermatitis',
    'Psoriasis pictures Lichen Planus and related diseases',
    'Scabies Lyme Disease and other Infestations and Bites',
    'Seborrheic Keratoses and other Benign Tumors',
    'Systemic Disease',
    'Tinea Ringworm Candidiasis and other Fungal Infections',
    'Urticaria Hives',
    'Vascular Tumors',
    'Vasculitis Photos',
    'Warts Molluscum and other Viral Infections'
]

# Mapping from technical folder names to exact clinical disease names
FRIENDLY_NAMES = {
    'Acne and Rosacea Photos': 'Acne & Rosacea',
    'Actinic Keratosis Basal Cell Carcinoma and other Malignant Lesions': 'Skin Cancer (Malignant)',
    'Atopic Dermatitis Photos': 'Atopic Dermatitis (Eczema)',
    'Bullous Disease Photos': 'Bullous Disease',
    'Cellulitis Impetigo and other Bacterial Infections': 'Bacterial Infection (Cellulitis/Impetigo)',
    'Eczema Photos': 'Eczema (Nonspecific)',
    'Exanthems and Drug Eruptions': 'Drug Eruptions & Exanthems',
    'Hair Loss Photos Alopecia and other Hair Diseases': 'Alopecia & Hair Loss',
    'Herpes HPV and other STDs Photos': 'Viral Infection (Herpes/HPV)',
    'Light Diseases and Disorders of Pigmentation': 'Pigmentation Disorders',
    'Lupus and other Connective Tissue diseases': 'Connective Tissue Disease (Lupus)',
    'Melanoma Skin Cancer Nevi and Moles': 'Melanoma & Moles',
    'Nail Fungus and other Nail Disease': 'Nail Fungus',
    'Poison Ivy Photos and other Contact Dermatitis': 'Contact Dermatitis (Poison Ivy)',
    'Psoriasis pictures Lichen Planus and related diseases': 'Psoriasis',
    'Scabies Lyme Disease and other Infestations and Bites': 'Infestations & Bites (Scabies/Lyme)',
    'Seborrheic Keratoses and other Benign Tumors': 'Benign Tumors (Seborrheic Keratoses)',
    'Systemic Disease': 'Systemic Skin Manifestations',
    'Tinea Ringworm Candidiasis and other Fungal Infections': 'Fungal Infection (Ringworm)',
    'Urticaria Hives': 'Urticaria (Hives)',
    'Vascular Tumors': 'Vascular Tumors',
    'Vasculitis Photos': 'Vasculitis',
    'Warts Molluscum and other Viral Infections': 'Viral Infections (Warts)'
}

# Mapping from technical names to IDs
FOLDER_TO_ID = {name: i for i, name in enumerate(CATEGORIES)}

# Medical reports template for new categories
REPORTS = {
    'Acne and Rosacea Photos': {
        "Description": "Acne is a skin condition that occurs when your hair follicles become plugged with oil and dead skin cells.",
        "Causes": "Excess oil production, clogged follicles, bacteria, and inflammation.",
        "Symptoms": "Whiteheads, blackheads, regular pimples, and painful cysts.",
        "Treatment": "Benzoyl peroxide, salicylic acid, retinoids, or antibiotics in severe cases."
    },
    'Eczema Photos': {
        "Description": "Eczema (dermatitis) refers to a group of conditions that cause skin to become itchy, dry, and inflamed.",
        "Causes": "Genetic factors, immune system overactivity, and environmental triggers.",
        "Symptoms": "Dry skin, severe itching, red patches, and small raised bumps.",
        "Treatment": "Moisturizers, topical corticosteroids, and avoiding triggers."
    },
    'Atopic Dermatitis Photos': {
        "Description": "Atopic Dermatitis is the most common form of Eczema, often starting in childhood.",
        "Causes": "Weakened skin barrier function combined with immune system triggers.",
        "Symptoms": "Chronic itchy rashes, often in the creases of the elbows or knees.",
        "Treatment": "Gentle skin care, steroid creams, and antihistamines for itching."
    },
    'Psoriasis pictures Lichen Planus and related diseases': {
        "Description": "Psoriasis is a disease in which skin cells build up and form scales and itchy, dry patches.",
        "Causes": "An immune system problem that causes skin cells to grow faster than usual.",
        "Symptoms": "Red patches of skin covered with thick, silvery scales; dry, cracked skin.",
        "Treatment": "Topical ointments, light therapy, and systemic medications."
    },
}

# Override REPORTS with full structured data for all 23 categories
REPORTS = {}
for cat in CATEGORIES:
    REPORTS[cat] = {
        'description': f'A dermatologist-confirmed diagnosis of {FRIENDLY_NAMES.get(cat, cat)} based on clinical image analysis.',
        'causes': ['Genetic predisposition', 'Environmental triggers', 'Immune system dysregulation', 'Skin barrier dysfunction'],
        'symptoms': ['Visible skin changes', 'Localised inflammation or redness', 'Itching or discomfort', 'Distinctive lesion patterns'],
        'treatment': ['Consult a board-certified dermatologist', 'Prescribed topical or systemic medications', 'Lifestyle and skincare modifications', 'Regular follow-up monitoring'],
        'prevention': ['Maintain good skin hygiene', 'Avoid known triggers', 'Use sunscreen and protective clothing', 'Stay hydrated and eat a balanced diet'],
        'duration': 'Varies depending on severity and treatment adherence',
        'when_to_see_doctor': ['If symptoms worsen or don\'t improve within 2 weeks', 'If the rash spreads rapidly', 'If fever or systemic symptoms develop', 'If you have a weakened immune system or diabetes'],
        'complications': []
    }

# Override specific categories with accurate clinical data
REPORTS['Acne and Rosacea Photos'].update({
    'description': 'Acne is a skin condition where hair follicles become clogged with oil and dead skin cells, causing pimples and cysts. Rosacea causes chronic facial redness and visible blood vessels.',
    'causes': ['Excess sebum production', 'Clogged hair follicles', 'Bacterial proliferation (C. acnes)', 'Hormonal fluctuations', 'Genetic predisposition'],
    'symptoms': ['Whiteheads and blackheads', 'Painful cysts and nodules', 'Facial redness and flushing', 'Visible blood vessels (rosacea)', 'Skin sensitivity'],
    'treatment': ['Benzoyl peroxide or salicylic acid washes', 'Topical retinoids', 'Antibiotics (topical or oral)', 'Isotretinoin for severe acne', 'Azelaic acid for rosacea'],
    'prevention': ['Gentle non-comedogenic skincare', 'Avoid touching your face', 'Use oil-free sunscreen daily', 'Identify and avoid rosacea triggers (heat, alcohol, spicy foods)'],
    'duration': 'Weeks to months; chronic for rosacea',
    'when_to_see_doctor': ['If OTC treatments fail after 8 weeks', 'If cysts appear or scarring begins', 'If the condition impacts mental health'],
    'complications': ['Permanent scarring', 'Post-inflammatory hyperpigmentation', 'Psychological distress']
})

REPORTS['Eczema Photos'].update({
    'description': 'Eczema (dermatitis) is a group of conditions causing itchy, inflamed, and dry skin due to a compromised skin barrier.',
    'causes': ['Impaired skin barrier function', 'Immune overactivation', 'Genetic mutations (filaggrin)', 'Environmental allergens', 'Stress'],
    'symptoms': ['Severe itching', 'Dry, scaly patches', 'Redness and swelling', 'Oozing or crusting in severe cases'],
    'treatment': ['Daily emollients and moisturisers', 'Topical corticosteroids', 'Calcineurin inhibitors (tacrolimus)', 'Antihistamines for itch', 'Biologic therapy (dupilumab) for severe cases'],
    'prevention': ['Use fragrance-free products', 'Moisturise immediately after bathing', 'Wear soft cotton clothing', 'Identify and avoid triggers'],
    'duration': 'Chronic with flare-ups; improves with consistent management',
    'when_to_see_doctor': ['If flares are frequent or severe', 'If sleep is disrupted by itching', 'If signs of skin infection appear'],
    'complications': ['Skin infection (impetigo)', 'Sleep disturbances', 'Increased asthma/allergy risk']
})

REPORTS['Psoriasis pictures Lichen Planus and related diseases'].update({
    'description': 'Psoriasis is a chronic autoimmune condition causing rapid skin cell proliferation. Lichen planus is an inflammatory condition affecting skin and mucous membranes.',
    'causes': ['Autoimmune T-cell activation', 'Genetic predisposition', 'Triggers: stress, infections, certain medications', 'Unknown cause for lichen planus'],
    'symptoms': ['Thick, silvery-scaled red plaques', 'Itching and burning', 'Flat-topped purple papules (lichen planus)', 'Nail changes', 'Joint pain (psoriatic arthritis)'],
    'treatment': ['Topical corticosteroids and vitamin D analogues', 'Phototherapy (UVB)', 'Methotrexate or cyclosporine', 'Biologic agents (TNF inhibitors)', 'Retinoids'],
    'prevention': ['Stress management', 'Avoid smoking and excess alcohol', 'Moisturise regularly', 'Report drug sensitivities to your doctor'],
    'duration': 'Chronic; remissions and flares are common',
    'when_to_see_doctor': ['If plaques cover large body areas', 'If joints become painful or swollen', 'If current treatments stop working'],
    'complications': ['Psoriatic arthritis', 'Cardiovascular risk', 'Depression and anxiety']
})
