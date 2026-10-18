import json
import os

TEST_SET = [
    # --- DIAGNOSTIC BRANDS (from requirements) ---
    {
        "id": "DIAG-01",
        "brand_name": "CampusWear",
        "business_description": "We sell trendy college apparel, hoodies, and sweatpants for students.",
        "industry": "fashion",
        "target_audience": "university students and young adults",
        "expected_category": "fashion",
        "category_hint": "fashion",
        "type": "diagnostic",
        "rationale": "Clear youth fashion apparel brand; primary product is collegiate casual clothing."
    },
    {
        "id": "DIAG-02",
        "brand_name": "GlowTheory",
        "business_description": "Clean, vegan skincare essentials for radiant and healthy skin.",
        "industry": "beauty & skincare",
        "target_audience": "skincare enthusiasts seeking a natural glow",
        "expected_category": "beauty",
        "category_hint": "beauty",
        "type": "diagnostic",
        "rationale": "Direct skincare and facial cosmetic formulations."
    },
    {
        "id": "DIAG-03",
        "brand_name": "FitForge",
        "business_description": "High-intensity functional training gear, barbells, and resistance bands.",
        "industry": "fitness equipment",
        "target_audience": "crossfit athletes and gym lifters",
        "expected_category": "fitness",
        "category_hint": "fitness",
        "type": "diagnostic",
        "rationale": "Weightlifting, workout equipment, and athletic gym training gear."
    },
    {
        "id": "DIAG-04",
        "brand_name": "SpiceRoute",
        "business_description": "Artisanal organic spice blends and small-batch curry pastes.",
        "industry": "gourmet food & cooking",
        "target_audience": "home cooks and foodies",
        "expected_category": "food",
        "category_hint": "food",
        "type": "diagnostic",
        "rationale": "Culinary seasonings and food preparation ingredients."
    },
    {
        "id": "DIAG-05",
        "brand_name": "UrbanNest",
        "business_description": "Mid-century modern home decor, handcrafted ceramics, and modular storage.",
        "industry": "interior design & furniture",
        "target_audience": "apartment renters and interior designers",
        "expected_category": "interior",
        "category_hint": "interior",
        "type": "diagnostic",
        "rationale": "Home furnishings, decor accents, and residential interior styling."
    },
    {
        "id": "DIAG-06",
        "brand_name": "PawPal",
        "business_description": "Hypoallergenic dog treats, durable chew toys, and ergonomic leashes.",
        "industry": "pet supplies",
        "target_audience": "dog owners and pet lovers",
        "expected_category": "pet",
        "category_hint": "pet",
        "type": "diagnostic",
        "rationale": "Canine nutrition, pet care accessories, and dog enrichment toys."
    },
    {
        "id": "DIAG-07",
        "brand_name": "WanderWeekends",
        "business_description": "Curated weekend road trip itineraries, glamping guides, and hiking gear rentals.",
        "industry": "travel & outdoor adventure",
        "target_audience": "weekend travelers and hikers",
        "expected_category": "travel",
        "category_hint": "travel",
        "type": "diagnostic",
        "rationale": "Leisure travel packages, itineraries, and outdoor tourism."
    },
    {
        "id": "DIAG-08",
        "brand_name": "CreatorHub",
        "business_description": "All-in-one link-in-bio, analytics, and invoicing platform for digital content creators.",
        "industry": "B2B SaaS / digital marketing",
        "target_audience": "social media creators and agencies",
        "expected_category": "out_of_domain",
        "category_hint": None,
        "type": "diagnostic",
        "rationale": "B2B SaaS utility; does not belong to consumer lifestyle pools."
    },

    # --- HISTORICAL NOTEBOOK 05 BRANDS (B1 to B10) ---
    {
        "id": "HIST-B1",
        "brand_name": "Brand_B1",
        "business_description": "Clean and organic skincare formulations",
        "industry": "skincare cosmetics",
        "target_audience": "women seeking natural beauty",
        "expected_category": "beauty",
        "category_hint": "beauty",
        "type": "clear",
        "rationale": "Historical test brand B1 from 05_brand_relevance.ipynb"
    },
    {
        "id": "HIST-B2",
        "brand_name": "Brand_B2",
        "business_description": "Matching outfits and gear for parents and toddlers",
        "industry": "family apparel",
        "target_audience": "young parents, moms, dads",
        "expected_category": "family",
        "category_hint": "family",
        "type": "clear",
        "rationale": "Historical test brand B2 from 05_brand_relevance.ipynb"
    },
    {
        "id": "HIST-B3",
        "brand_name": "Brand_B3",
        "business_description": "Trendy streetwear and oversized hoodies",
        "industry": "fashion apparel",
        "target_audience": "teens and young adults",
        "expected_category": "fashion",
        "category_hint": "fashion",
        "type": "clear",
        "rationale": "Historical test brand B3 from 05_brand_relevance.ipynb"
    },
    {
        "id": "HIST-B4",
        "brand_name": "Brand_B4",
        "business_description": "High-protein meal replacement shakes",
        "industry": "fitness supplements",
        "target_audience": "gym goers, athletes",
        "expected_category": "fitness",
        "category_hint": "fitness",
        "type": "clear",
        "rationale": "Historical test brand B4 from 05_brand_relevance.ipynb"
    },
    {
        "id": "HIST-B5",
        "brand_name": "Brand_B5",
        "business_description": "Gourmet meal kit delivery service",
        "industry": "food delivery",
        "target_audience": "busy professionals, home cooks",
        "expected_category": "food",
        "category_hint": "food",
        "type": "clear",
        "rationale": "Historical test brand B5 from 05_brand_relevance.ipynb"
    },
    {
        "id": "HIST-B6",
        "brand_name": "Brand_B6",
        "business_description": "Minimalist scandinavian living room furniture",
        "industry": "home furniture",
        "target_audience": "homeowners, decorators",
        "expected_category": "interior",
        "category_hint": "interior",
        "type": "clear",
        "rationale": "Historical test brand B6 from 05_brand_relevance.ipynb"
    },
    {
        "id": "HIST-B7",
        "brand_name": "Brand_B7",
        "business_description": "Grain-free premium dog kibble",
        "industry": "pet food",
        "target_audience": "dog owners",
        "expected_category": "pet",
        "category_hint": "pet",
        "type": "clear",
        "rationale": "Historical test brand B7 from 05_brand_relevance.ipynb"
    },
    {
        "id": "HIST-B8",
        "brand_name": "Brand_B8",
        "business_description": "Guided backpacking tours in the alps",
        "industry": "travel tourism",
        "target_audience": "adventure seekers, hikers",
        "expected_category": "travel",
        "category_hint": "travel",
        "type": "clear",
        "rationale": "Historical test brand B8 from 05_brand_relevance.ipynb"
    },
    {
        "id": "HIST-B9",
        "brand_name": "Brand_B9",
        "business_description": "Enterprise cloud accounting software",
        "industry": "B2B SaaS",
        "target_audience": "CFOs, accountants",
        "expected_category": "out_of_domain",
        "category_hint": None,
        "type": "out_of_domain",
        "rationale": "Historical test brand B9 from 05_brand_relevance.ipynb"
    },
    {
        "id": "HIST-B10",
        "brand_name": "Brand_B10",
        "business_description": "Network penetration testing services",
        "industry": "cybersecurity",
        "target_audience": "IT administrators",
        "expected_category": "out_of_domain",
        "category_hint": None,
        "type": "out_of_domain",
        "rationale": "Historical test brand B10 from 05_brand_relevance.ipynb"
    },

    # --- CLEAR / EASY IN-DOMAIN EXAMPLES ---
    {
        "id": "CLEAR-01",
        "brand_name": "ChicBoutique",
        "business_description": "Designer evening dresses, silk blouses, and luxury accessories.",
        "industry": "apparel",
        "target_audience": "fashion lovers",
        "expected_category": "fashion",
        "category_hint": "fashion",
        "type": "clear",
        "rationale": "Unambiguous women's formal fashion apparel."
    },
    {
        "id": "CLEAR-02",
        "brand_name": "VelvetSkin",
        "business_description": "Hydrating hyaluronic acid serums, anti-aging moisturizers, and facial oils.",
        "industry": "cosmetics",
        "target_audience": "women 25-50",
        "expected_category": "beauty",
        "category_hint": "beauty",
        "type": "clear",
        "rationale": "Unambiguous cosmetic skincare."
    },
    {
        "id": "CLEAR-03",
        "brand_name": "IronPulse",
        "business_description": "Pre-workout energy supplements and whey protein isolate powders.",
        "industry": "sports nutrition",
        "target_audience": "bodybuilders",
        "expected_category": "fitness",
        "category_hint": "fitness",
        "type": "clear",
        "rationale": "Athletic fitness supplements."
    },
    {
        "id": "CLEAR-04",
        "brand_name": "SweetBake",
        "business_description": "Freshly baked sourdough bread, artisanal pastries, and custom celebration cakes.",
        "industry": "bakery",
        "target_audience": "dessert lovers",
        "expected_category": "food",
        "category_hint": "food",
        "type": "clear",
        "rationale": "Unambiguous bakery and food products."
    },
    {
        "id": "CLEAR-05",
        "brand_name": "HavenLiving",
        "business_description": "Scandinavian living room styling, acoustic wall panels, and minimalist lighting.",
        "industry": "home decor",
        "target_audience": "homeowners",
        "expected_category": "interior",
        "category_hint": "interior",
        "type": "clear",
        "rationale": "Unambiguous residential interior design."
    },
    {
        "id": "CLEAR-06",
        "brand_name": "FelineCo",
        "business_description": "Grain-free cat food, ergonomic scratching posts, and interactive laser toys.",
        "industry": "cat care",
        "target_audience": "cat parents",
        "expected_category": "pet",
        "category_hint": "pet",
        "type": "clear",
        "rationale": "Pet supplies specifically for cats."
    },
    {
        "id": "CLEAR-07",
        "brand_name": "AlpineTrek",
        "business_description": "Guided mountaineering expeditions and summit trek packages in the Rocky Mountains.",
        "industry": "travel tourism",
        "target_audience": "mountaineers and backpackers",
        "expected_category": "travel",
        "category_hint": "travel",
        "type": "clear",
        "rationale": "Adventure tourism and travel expeditions."
    },
    {
        "id": "CLEAR-08",
        "brand_name": "LittleSprouts",
        "business_description": "Organic cotton baby onesies, sensory developmental toys, and toddler feeding sets.",
        "industry": "baby & parenting",
        "target_audience": "new moms and toddlers",
        "expected_category": "family",
        "category_hint": "family",
        "type": "clear",
        "rationale": "Parenting, infant supplies, and family life."
    },

    # --- SHORT DESCRIPTIONS (Sparse token challenge) ---
    {
        "id": "SHORT-01",
        "brand_name": "ModCloth",
        "business_description": "Vintage thrifted streetwear.",
        "industry": "fashion",
        "target_audience": "youth",
        "expected_category": "fashion",
        "category_hint": "fashion",
        "type": "short",
        "rationale": "Ultra-short fashion query testing sparse semantic capture."
    },
    {
        "id": "SHORT-02",
        "brand_name": "PureGlow",
        "business_description": "Organic lip gloss.",
        "industry": "beauty",
        "target_audience": "teens",
        "expected_category": "beauty",
        "category_hint": "beauty",
        "type": "short",
        "rationale": "Ultra-short beauty cosmetic query."
    },
    {
        "id": "SHORT-03",
        "brand_name": "CoreGym",
        "business_description": "Kettlebell workouts.",
        "industry": "fitness",
        "target_audience": "athletes",
        "expected_category": "fitness",
        "category_hint": "fitness",
        "type": "short",
        "rationale": "Ultra-short workout fitness query."
    },
    {
        "id": "SHORT-04",
        "brand_name": "TacoCraft",
        "business_description": "Street tacos and salsa.",
        "industry": "food",
        "target_audience": "foodies",
        "expected_category": "food",
        "category_hint": "food",
        "type": "short",
        "rationale": "Ultra-short culinary query."
    },
    {
        "id": "SHORT-05",
        "brand_name": "CozyCasa",
        "business_description": "Boho rug collection.",
        "industry": "interior",
        "target_audience": "renters",
        "expected_category": "interior",
        "category_hint": "interior",
        "type": "short",
        "rationale": "Ultra-short home furnishing query."
    },
    {
        "id": "SHORT-06",
        "brand_name": "BarkBite",
        "business_description": "Raw beef dog food.",
        "industry": "pet",
        "target_audience": "dogs",
        "expected_category": "pet",
        "category_hint": "pet",
        "type": "short",
        "rationale": "Ultra-short pet nutrition query."
    },
    {
        "id": "SHORT-07",
        "brand_name": "NomadRoam",
        "business_description": "Solo budget hostels.",
        "industry": "travel",
        "target_audience": "backpackers",
        "expected_category": "travel",
        "category_hint": "travel",
        "type": "short",
        "rationale": "Ultra-short budget travel query."
    },
    {
        "id": "SHORT-08",
        "brand_name": "TinyTots",
        "business_description": "Strollers and baby carriers.",
        "industry": "family",
        "target_audience": "parents",
        "expected_category": "family",
        "category_hint": "family",
        "type": "short",
        "rationale": "Ultra-short nursery and parenting equipment query."
    },

    # --- LONG DESCRIPTIONS (Rich Context) ---
    {
        "id": "LONG-01",
        "brand_name": "TailorAndThread",
        "business_description": "Bespoke menswear tailoring atelier crafting made-to-measure Italian wool suits, formal dress shirts, cashmere overcoats, and handcrafted leather loafers for discerning gentlemen and business professionals.",
        "industry": "luxury menswear fashion",
        "target_audience": "executive professionals and wedding parties",
        "expected_category": "fashion",
        "category_hint": "fashion",
        "type": "long",
        "rationale": "Elaborate multi-sentence luxury sartorial description."
    },
    {
        "id": "LONG-02",
        "brand_name": "BotanicalBeautyLab",
        "business_description": "Dermatologist-formulated clinical skincare powered by bioactive marine botanicals, niacinamide, and stabilized vitamin C, clinically proven to repair skin barriers and enhance skin elasticity.",
        "industry": "clinical skincare & beauty",
        "target_audience": "consumers with sensitive or aging skin",
        "expected_category": "beauty",
        "category_hint": "beauty",
        "type": "long",
        "rationale": "Elaborate dermatological formulation description."
    },
    {
        "id": "LONG-03",
        "brand_name": "ApexStrengthSystems",
        "business_description": "Comprehensive athletic performance coaching combining Olympic weightlifting programming, biomechanical barbell analysis, metabolic conditioning, and recovery protocols.",
        "industry": "strength & conditioning",
        "target_audience": "competitive powerlifters and athletes",
        "expected_category": "fitness",
        "category_hint": "fitness",
        "type": "long",
        "rationale": "Elaborate strength sports and coaching description."
    },
    {
        "id": "LONG-04",
        "brand_name": "FarmhouseFeasts",
        "business_description": "Farm-to-table organic meal subscription featuring locally sourced seasonal produce, grass-fed meats, heirloom grains, and chef-curated recipes delivered weekly.",
        "industry": "gourmet food subscription",
        "target_audience": "health-conscious families and culinary enthusiasts",
        "expected_category": "food",
        "category_hint": "food",
        "type": "long",
        "rationale": "Elaborate culinary meal service description."
    },

    # --- AMBIGUOUS / BOUNDARY CASES ---
    {
        "id": "AMBIG-01",
        "brand_name": "AthleisureChic",
        "business_description": "High-performance compression leggings and stylish crop tops designed for transition from morning yoga sessions to casual downtown brunch.",
        "industry": "activewear apparel",
        "target_audience": "active women who value style",
        "expected_category": "fashion",
        "category_hint": "fashion",
        "type": "ambiguous",
        "rationale": "Apparel focus with fitness cross-utility; intended primary category is fashion apparel."
    },
    {
        "id": "AMBIG-02",
        "brand_name": "MomFitLife",
        "business_description": "Postpartum rehabilitation fitness courses, core recovery workouts, and pelvic health guides for new mothers.",
        "industry": "maternal fitness",
        "target_audience": "postpartum moms",
        "expected_category": "fitness",
        "category_hint": "fitness",
        "type": "ambiguous",
        "rationale": "Physical exercise workouts targeted at mothers; primary domain is fitness."
    },
    {
        "id": "AMBIG-03",
        "brand_name": "TravelBoutique",
        "business_description": "Lightweight, wrinkle-free travel clothing, packable linen shirts, and convertible travel bags for global explorers.",
        "industry": "travel apparel",
        "target_audience": "frequent flyers and nomad travelers",
        "expected_category": "fashion",
        "category_hint": "fashion",
        "type": "ambiguous",
        "rationale": "Wearable garments designed for tourism; core category is fashion apparel."
    },
    {
        "id": "AMBIG-04",
        "brand_name": "PetHavenCafe",
        "business_description": "Pet-friendly neighborhood coffee house serving specialty espresso and artisan pastries alongside freshly baked dog treats.",
        "industry": "hospitality & cafe",
        "target_audience": "pet owners and coffee enthusiasts",
        "expected_category": "food",
        "category_hint": "food",
        "type": "ambiguous",
        "rationale": "Food/hospitality cafe venue accommodating animal companions."
    },
    {
        "id": "AMBIG-05",
        "brand_name": "NurseryNook",
        "business_description": "Sustainable Montessori wooden nursery furniture, cribs, and baby room interior decor.",
        "industry": "nursery design",
        "target_audience": "expecting parents",
        "expected_category": "interior",
        "category_hint": "interior",
        "type": "ambiguous",
        "rationale": "Furniture and spatial room design tailored for infants; core domain is interior decor."
    },

    # --- ADVERSARIAL / POLYSEMOUS SEMANTIC CASES ---
    {
        "id": "ADV-01",
        "brand_name": "AppleCoreTech",
        "business_description": "Refurbished Apple MacBooks, iPhone repair kits, and hardware diagnostic tools.",
        "industry": "consumer electronics",
        "target_audience": "tech enthusiasts",
        "expected_category": "out_of_domain",
        "category_hint": None,
        "type": "adversarial",
        "rationale": "Polysemous token 'Apple' must not resolve to food; tech hardware is out-of-domain."
    },
    {
        "id": "ADV-02",
        "brand_name": "RunningABusiness",
        "business_description": "Executive coaching, cash flow optimization, and leadership workshops for entrepreneurs.",
        "industry": "business consulting",
        "target_audience": "small business owners",
        "expected_category": "out_of_domain",
        "category_hint": None,
        "type": "adversarial",
        "rationale": "Polysemous verb 'Running' must not resolve to fitness; business consulting is out-of-domain."
    },
    {
        "id": "ADV-03",
        "brand_name": "CookingTheBooks",
        "business_description": "Forensic accounting, corporate fraud detection, and financial auditing services.",
        "industry": "accounting & finance",
        "target_audience": "corporate boards and legal teams",
        "expected_category": "out_of_domain",
        "category_hint": None,
        "type": "adversarial",
        "rationale": "Idiom 'Cooking' must not resolve to culinary food; financial auditing is out-of-domain."
    },
    {
        "id": "ADV-04",
        "brand_name": "PetProjectStudio",
        "business_description": "Independent indie game development studio building 2D pixel-art roguelike RPGs.",
        "industry": "video games",
        "target_audience": "PC gamers",
        "expected_category": "out_of_domain",
        "category_hint": None,
        "type": "adversarial",
        "rationale": "Idiom 'Pet project' must not resolve to animals/pet; video gaming is out-of-domain."
    },
    {
        "id": "ADV-05",
        "brand_name": "OrganicGrowthAgency",
        "business_description": "Search engine optimization, viral content strategy, and organic growth marketing for B2B tech startups.",
        "industry": "digital marketing",
        "target_audience": "startup founders",
        "expected_category": "out_of_domain",
        "category_hint": None,
        "type": "adversarial",
        "rationale": "Marketing term 'Organic' must not resolve to food/beauty; digital marketing is out-of-domain."
    },
    {
        "id": "ADV-06",
        "brand_name": "HotDogMedia",
        "business_description": "Podcasting studio and audio engineering services for narrative audio journalists.",
        "industry": "media production",
        "target_audience": "podcasters",
        "expected_category": "out_of_domain",
        "category_hint": None,
        "type": "adversarial",
        "rationale": "Brand name 'HotDog' must not resolve to food or pet; audio media is out-of-domain."
    },
    {
        "id": "ADV-07",
        "brand_name": "FitnessTrackerApp",
        "business_description": "GPS route mapping and social leaderboard mobile app for cycling clubs and runners.",
        "industry": "mobile software",
        "target_audience": "cyclists and endurance runners",
        "expected_category": "fitness",
        "category_hint": "fitness",
        "type": "adversarial",
        "rationale": "Software application whose true semantic domain is athletic fitness and running."
    },

    # --- PURE OUT-OF-DOMAIN CASES ---
    {
        "id": "OOD-01",
        "brand_name": "CyberShield",
        "business_description": "Zero-trust network architecture, penetration testing, and endpoint security for healthcare providers.",
        "industry": "cybersecurity",
        "target_audience": "CISOs and IT managers",
        "expected_category": "out_of_domain",
        "category_hint": None,
        "type": "out_of_domain",
        "rationale": "Enterprise cybersecurity software; completely outside 8 lifestyle pools."
    },
    {
        "id": "OOD-02",
        "brand_name": "ApexCloudSaaS",
        "business_description": "Automated container orchestration, Kubernetes cluster management, and CI/CD deployment pipelines.",
        "industry": "cloud infrastructure",
        "target_audience": "DevOps engineers",
        "expected_category": "out_of_domain",
        "category_hint": None,
        "type": "out_of_domain",
        "rationale": "DevOps cloud computing infrastructure; completely outside 8 lifestyle pools."
    },
    {
        "id": "OOD-03",
        "brand_name": "LexLegalConsult",
        "business_description": "Corporate restructuring, intellectual property litigation, and cross-border commercial arbitration.",
        "industry": "legal services",
        "target_audience": "multinational corporations",
        "expected_category": "out_of_domain",
        "category_hint": None,
        "type": "out_of_domain",
        "rationale": "Corporate law firm; completely outside 8 lifestyle pools."
    },
    {
        "id": "OOD-04",
        "brand_name": "HeavyMachineryDirect",
        "business_description": "Hydraulic excavators, bulldozers, and industrial cranes for civil construction projects.",
        "industry": "construction equipment",
        "target_audience": "contractors and fleet managers",
        "expected_category": "out_of_domain",
        "category_hint": None,
        "type": "out_of_domain",
        "rationale": "Heavy industrial civil machinery; completely outside 8 lifestyle pools."
    }
]

out_path = os.path.join(os.path.dirname(__file__), "data", "relevance_test_set.json")
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(TEST_SET, f, indent=2)

print(f"Generated {len(TEST_SET)} test examples in {out_path}.")
