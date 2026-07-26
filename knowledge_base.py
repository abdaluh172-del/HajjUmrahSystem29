# -*- coding: utf-8 -*-
"""Curated Hajj & Umrah knowledge base (v15).

This is NOT a substitute for official sources — it is a small, carefully
written set of grounding facts the assistant (assistant.py) can lean on so
it answers from something concrete instead of guessing. Every entry stays
at a level of detail that is well-established and non-controversial; where
scholars differ on details, the entry says so explicitly and points the
person to an official Ifta/guidance office instead of picking a side.

Each entry:
    {
        "id": "ihram",
        "keywords_ar": [...], "keywords_en": [...],   # for lightweight retrieval
        "title_ar": "...", "title_en": "...",
        "body_ar": "...",                              # short, structured, Arabic
        "body_en": "...",                               # short, structured, English
    }

retrieve(query, limit=3) does simple keyword overlap scoring — no embeddings,
no network calls, so it works even with zero configuration. It is meant to
be injected into the LLM's context (grounding) AND used directly as a
template-based fallback answer when no LLM key is configured.
"""
import re

OFFICIAL_SOURCES_AR = (
    "الهيئة العامة للعناية بشؤون المسجد الحرام والمسجد النبوي، "
    "وزارة الحج والعمرة، منصة نسك (Nusuk)، ورئاسة الشؤون الدينية "
    "بالمسجدين الشريفين"
)
OFFICIAL_SOURCES_EN = (
    "the General Presidency for the Affairs of the Grand Mosque and the "
    "Prophet's Mosque, the Saudi Ministry of Hajj and Umrah, the Nusuk "
    "platform, and the Presidency of Religious Affairs at the Two Holy Mosques"
)

KB = [
    {
        "id": "ihram",
        "keywords_ar": ["احرام", "إحرام", "ميقات", "مواقيت", "نية", "تلبية", "محظورات"],
        "keywords_en": ["ihram", "miqat", "meeqat", "intention", "talbiyah", "prohibitions"],
        "title_ar": "الإحرام والمواقيت",
        "title_en": "Ihram and the Miqats",
        "body_ar": (
            "- الإحرام هو نية الدخول في نسك الحج أو العمرة، ويكون من أحد المواقيت المكانية "
            "المحددة (مثل ذو الحليفة لأهل المدينة، والجحفة لأهل الشام، وقرن المنازل لأهل نجد، "
            "ويلملم لأهل اليمن، وذات عرق لأهل العراق)، أو من مكان الإقامة لمن كان داخل حدود المواقيت.\n"
            "- يُستحب الاغتسال ولبس ملابس الإحرام (للرجال إزار ورداء أبيضان)، ثم عقد النية والتلبية: "
            "«لبيك اللهم لبيك، لبيك لا شريك لك لبيك...».\n"
            "- من محظورات الإحرام: تغطية الرأس للرجل، لبس المخيط للرجل، تقليم الأظافر، حلق الشعر، "
            "استخدام الطيب بعد النية، عقد النكاح، الصيد، وقطع شجر الحرم.\n"
            "- من ارتكب محظورًا لعذر أو بغيره فقد تلزمه فدية — تفاصيل الفدية تختلف بحسب نوع المحظور، "
            "والأولى سؤال مكتب الإفتاء أو المرشدين الرسميين عن الحالة الخاصة."
        ),
        "body_en": (
            "- Ihram is the intention to begin the rites of Hajj or Umrah, entered into at one of "
            "the designated Miqat boundaries (e.g. Dhul-Hulayfah, Al-Juhfah, Qarn al-Manazil, "
            "Yalamlam, Dhat 'Irq), or from one's place of residence if already inside the Miqat zone.\n"
            "- It is recommended to perform ghusl and wear ihram garments (for men: two white "
            "unstitched cloths), then form the intention and recite the Talbiyah.\n"
            "- Prohibitions during Ihram include: covering the head (men), wearing stitched/tailored "
            "clothing (men), cutting nails, cutting hair, using perfume after the intention, marriage "
            "contracts, hunting, and cutting plants within the sanctuary.\n"
            "- Violating a prohibition may require an expiation (fidyah); the specific ruling depends "
            "on the situation — it's best to ask an official Ifta office or accredited guide."
        ),
    },
    {
        "id": "tawaf",
        "keywords_ar": ["طواف", "الطواف", "المطاف", "اضطباع", "رمل", "الحجر الأسود"],
        "keywords_en": ["tawaf", "circumambulation", "black stone", "matalaf"],
        "title_ar": "الطواف",
        "title_en": "Tawaf",
        "body_ar": (
            "- الطواف هو الدوران حول الكعبة المشرفة سبعة أشواط، يبدأ وينتهي عند الحجر الأسود، "
            "وتكون الكعبة عن يسار الطائف.\n"
            "- يُسن للرجال في طواف القدوم الاضطباع (كشف الكتف الأيمن) والرمل (الإسراع بخطى "
            "متقاربة) في الأشواط الثلاثة الأولى فقط.\n"
            "- تُستحب الإشارة إلى الحجر الأسود أو استلامه إن أمكن دون مزاحمة، مع التكبير.\n"
            "- بعد الطواف يُستحب صلاة ركعتين خلف مقام إبراهيم إن تيسر، وإلا ففي أي مكان بالمسجد.\n"
            "- أنواع الطواف تشمل: طواف القدوم، طواف الإفاضة (ركن من أركان الحج)، وطواف الوداع "
            "(واجب لمن غادر مكة)."
        ),
        "body_en": (
            "- Tawaf is circling the Ka'bah seven times, starting and ending at the Black Stone, "
            "keeping the Ka'bah on the left.\n"
            "- In Tawaf al-Qudum, men perform Idtiba' (uncovering the right shoulder) and Raml "
            "(brisk short steps) for the first three circuits only.\n"
            "- It's recommended to point toward or touch the Black Stone if possible without "
            "pushing through crowds, saying the takbeer.\n"
            "- Afterward, praying two rak'ahs behind Maqam Ibrahim (or anywhere in the mosque if "
            "crowded) is recommended.\n"
            "- Types include: Tawaf al-Qudum (arrival), Tawaf al-Ifadah (a pillar of Hajj), and "
            "Tawaf al-Wada' (farewell, obligatory before leaving Makkah)."
        ),
    },
    {
        "id": "sai",
        "keywords_ar": ["سعي", "السعي", "المسعى", "الصفا", "المروة"],
        "keywords_en": ["sai", "safa", "marwa", "marwah"],
        "title_ar": "السعي بين الصفا والمروة",
        "title_en": "Sa'i between Safa and Marwah",
        "body_ar": (
            "- السعي هو المشي بين جبلي الصفا والمروة سبعة أشواط، يبدأ بالصفا وينتهي بالمروة.\n"
            "- يُستحب للرجال الإسراع (الهرولة) بين العلمين الأخضرين، أما النساء فيمشين بخطى عادية.\n"
            "- يكون السعي بعد طواف القدوم في العمرة، أو بعد طواف الإفاضة في الحج (أو بعد طواف "
            "القدوم لمن قدّمه، حسب نوع النسك: تمتع أو قِران أو إفراد)."
        ),
        "body_en": (
            "- Sa'i is walking between the hills of Safa and Marwah seven times, starting at Safa "
            "and ending at Marwah.\n"
            "- Men are recommended to jog briskly between the two green markers; women walk at a "
            "normal pace throughout.\n"
            "- Sa'i follows Tawaf al-Qudum in Umrah, or Tawaf al-Ifadah in Hajj (timing can vary "
            "slightly by the type of Hajj performed: Tamattu', Qiran, or Ifrad)."
        ),
    },
    {
        "id": "arafah",
        "keywords_ar": ["عرفة", "الوقوف بعرفة", "يوم عرفة", "جبل الرحمة"],
        "keywords_en": ["arafah", "arafat", "day of arafah", "mount of mercy"],
        "title_ar": "الوقوف بعرفة",
        "title_en": "The Standing at Arafah",
        "body_ar": (
            "- الوقوف بعرفة يوم 9 ذي الحجة هو ركن الحج الأعظم؛ من فاته فقد فاته الحج.\n"
            "- يمتد وقته من زوال شمس يوم عرفة إلى فجر يوم النحر، والأفضل الوقوف حتى الغروب.\n"
            "- يُستحب الإكثار من الدعاء والذكر والاستغفار وقراءة القرآن في هذا اليوم."
        ),
        "body_en": (
            "- Standing at Arafah on 9 Dhul-Hijjah is the greatest pillar of Hajj; missing it means "
            "missing Hajj that year.\n"
            "- Its time spans from midday on the 9th until dawn on the 10th (the Day of Sacrifice); "
            "it's best to stay until sunset.\n"
            "- It's recommended to devote the day to supplication, remembrance, seeking forgiveness "
            "and reciting the Qur'an."
        ),
    },
    {
        "id": "muzdalifah_mina",
        "keywords_ar": ["مزدلفة", "منى", "المبيت", "جمع"],
        "keywords_en": ["muzdalifah", "mina", "overnight stay"],
        "title_ar": "مزدلفة ومنى",
        "title_en": "Muzdalifah and Mina",
        "body_ar": (
            "- بعد الإفاضة من عرفة يتوجه الحجاج إلى مزدلفة للمبيت فيها ليلة العيد، ويُجمع بها "
            "المغرب والعشاء، ويُلتقط منها حصى الجمرات.\n"
            "- يُرخَّص لأصحاب الأعذار (كبار السن، المرضى، ومن يرافقهم) بالدفع من مزدلفة بعد "
            "منتصف الليل.\n"
            "- منى هي مكان رمي الجمرات والمبيت أيام التشريق (11، 12، وأحيانًا 13 ذي الحجة)."
        ),
        "body_en": (
            "- After leaving Arafah, pilgrims head to Muzdalifah to spend the night, combining "
            "Maghrib and Isha prayers there, and collecting pebbles for the Jamarat.\n"
            "- Those with valid excuses (elderly, sick, and their companions) may leave Muzdalifah "
            "after midnight.\n"
            "- Mina is where the Jamarat are stoned and where pilgrims stay during the Tashreeq days "
            "(11th, 12th, and sometimes 13th of Dhul-Hijjah)."
        ),
    },
    {
        "id": "jamarat_hady",
        "keywords_ar": ["رمي الجمرات", "الجمرات", "الهدي", "الأضحية", "حلق", "تقصير"],
        "keywords_en": ["jamarat", "stoning", "hady", "sacrifice", "shaving", "trimming"],
        "title_ar": "رمي الجمرات، الهدي، والحلق أو التقصير",
        "title_en": "Stoning the Jamarat, the Sacrifice, and Shaving/Trimming",
        "body_ar": (
            "- يوم النحر (10 ذي الحجة): يُرمى جمرة العقبة الكبرى بسبع حصيات، ثم يُذبح الهدي "
            "(لمن لزمه)، ثم يُحلق الرأس أو يُقصَّر (الحلق أفضل للرجال)، ثم طواف الإفاضة.\n"
            "- أيام التشريق (11-12، وأحيانًا 13): تُرمى الجمرات الثلاث (الصغرى، الوسطى، الكبرى) "
            "كل يوم بسبع حصيات لكل جمرة، بعد الزوال.\n"
            "- يمكن التعجل بمغادرة منى بعد رمي اليوم الثاني عشر قبل الغروب، أو التأخر لليوم الثالث "
            "عشر."
        ),
        "body_en": (
            "- Day of Sacrifice (10 Dhul-Hijjah): stone the largest Jamrah (Al-'Aqabah) with seven "
            "pebbles, offer the sacrifice (if obligatory), then shave the head or trim the hair "
            "(shaving is preferred for men), then perform Tawaf al-Ifadah.\n"
            "- Tashreeq days (11th-12th, sometimes 13th): stone all three Jamarat (small, middle, "
            "large) with seven pebbles each, after midday.\n"
            "- Pilgrims may leave Mina early after the 12th's stoning (before sunset) or stay through "
            "the 13th."
        ),
    },
    {
        "id": "miqat_fidyah",
        "keywords_ar": ["فدية", "دم", "كفارة"],
        "keywords_en": ["fidyah", "expiation", "penalty"],
        "title_ar": "الفدية",
        "title_en": "Fidyah (Expiation)",
        "body_ar": (
            "- الفدية تجب في حالات مختلفة (مثل حلق الشعر لعذر، أو ترك واجب) وتُقدَّر بصيام ثلاثة "
            "أيام، أو إطعام ستة مساكين، أو ذبح شاة — والتفاصيل تختلف حسب سبب الفدية.\n"
            "- لتحديد الحكم الدقيق لحالتك، يُفضَّل التواصل مع مكاتب الإفتاء الرسمية المتوفرة في "
            "المشاعر المقدسة والحرمين."
        ),
        "body_en": (
            "- Fidyah applies in various situations (e.g. excused hair removal, or omitting an "
            "obligatory act) and is typically fasting three days, feeding six poor people, or "
            "sacrificing a sheep — the exact form depends on the reason.\n"
            "- For a precise ruling on your specific situation, it's best to contact the official "
            "Ifta offices available at the holy sites and the two mosques."
        ),
    },
    {
        "id": "haramain_services",
        "keywords_ar": [
            "خدمات المسجد الحرام", "خدمات المسجد النبوي", "دورات المياه", "أماكن الوضوء",
            "البوابات", "المصليات", "عربات كبار السن", "عربات ذوي الإعاقة", "الإسعافات الأولية",
            "المراكز الصحية", "المفقودات", "تنظيم الحشود", "خدمات التنقل",
        ],
        "keywords_en": [
            "grand mosque services", "prophet's mosque services", "restrooms", "ablution",
            "gates", "prayer areas", "elderly carts", "wheelchair", "first aid", "health center",
            "lost and found", "crowd control", "shuttle", "transport services",
        ],
        "title_ar": "خدمات الحرمين الشريفين",
        "title_en": "Services at the Two Holy Mosques",
        "body_ar": (
            "- يوفر الحرمان الشريفان دورات مياه وأماكن وضوء منتشرة عند معظم البوابات ومناطق "
            "التوسعة.\n"
            "- تتوفر عربات كهربائية لكبار السن وذوي الإعاقة للتنقل داخل ساحات الحرمين — تُطلب "
            "عادة من نقاط الخدمة المخصصة أو عبر المتطوعين والمرشدين.\n"
            "- توجد نقاط إسعافات أولية ومراكز صحية داخل الحرمين وفي محيطهما لحالات الطوارئ.\n"
            "- لخدمة المفقودات (أشخاص أو أغراض) هناك مكاتب مخصصة يمكن سؤال أقرب رجل أمن أو "
            "متطوع عن موقعها.\n"
            "- بوابات الحرمين مرقّمة ومُعلَّمة لتسهيل تحديد المكان والالتقاء؛ يُنصح بحفظ اسم أو "
            "رقم البوابة الأقرب لمكان إقامتك.\n"
            "- التطبيقات الرسمية (مثل تطبيق نسك، وتطبيقات الهيئة العامة للعناية بشؤون المسجدين) "
            "توفر معلومات محدثة عن الازدحام، الفعاليات، ومواعيد الصلاة."
        ),
        "body_en": (
            "- Restrooms and ablution areas are distributed near most gates and expansion zones of "
            "both mosques.\n"
            "- Electric carts for elderly and disabled visitors are available for moving within the "
            "mosque courtyards — usually requested at dedicated service points or from volunteers "
            "and guides.\n"
            "- First-aid points and health centers are located inside and around both mosques for "
            "emergencies.\n"
            "- Dedicated lost-and-found offices (for people or belongings) exist — ask the nearest "
            "security officer or volunteer for the location.\n"
            "- Gates at both mosques are numbered and signed to help with navigation and meeting "
            "points; it helps to remember the gate number nearest your accommodation.\n"
            "- Official apps (such as Nusuk and the General Presidency's apps) provide up-to-date "
            "information on crowding, events, and prayer times."
        ),
    },
    {
        "id": "pillars",
        "keywords_ar": [
            "أركان الحج", "أركان العمرة", "اركان الحج", "اركان العمرة", "ركن الحج",
            "ركن العمرة", "انواع الحج", "أنواع الحج", "تمتع", "قران", "إفراد",
        ],
        "keywords_en": [
            "pillars of hajj", "pillars of umrah", "pillar of hajj", "pillar of umrah",
            "types of hajj", "tamattu", "qiran", "ifrad",
        ],
        "title_ar": "أركان الحج والعمرة (نظرة عامة)",
        "title_en": "Pillars of Hajj and Umrah (Overview)",
        "body_ar": (
            "- أركان العمرة: الإحرام (نية الدخول في النسك)، الطواف بالكعبة، والسعي بين الصفا "
            "والمروة. من ترك ركنًا لم تصح عمرته حتى يأتي به.\n"
            "- أركان الحج المتفق عليها بين جمهور الفقهاء: الإحرام، الوقوف بعرفة (وهو الركن "
            "الأعظم)، طواف الإفاضة، والسعي بين الصفا والمروة.\n"
            "- تفاصيل عدد الأركان بالضبط، وحكم الحلق/التقصير (ركن أم واجب)، تختلف قليلًا بين "
            "المذاهب الفقهية — يُنصح بمراجعة مكتب إفتاء رسمي لمعرفة الرأي المعتمد في حالتك.\n"
            "- أنواع النسك الثلاثة (التمتع، القِران، الإفراد) تحدد ترتيب أداء العمرة والحج "
            "ووجوب الهدي؛ يختار الحاج نوعه عند الإحرام."
        ),
        "body_en": (
            "- Pillars of Umrah: Ihram (the intention), Tawaf around the Ka'bah, and Sa'i "
            "between Safa and Marwah. Omitting a pillar means the Umrah isn't valid until it's "
            "performed.\n"
            "- Pillars of Hajj agreed upon by the majority of scholars: Ihram, standing at "
            "Arafah (the greatest pillar), Tawaf al-Ifadah, and Sa'i between Safa and Marwah.\n"
            "- The exact count and whether shaving/trimming counts as a pillar or an obligation "
            "varies slightly between schools of jurisprudence — check with an official Ifta "
            "office for the view that applies to your situation.\n"
            "- The three types of pilgrimage (Tamattu', Qiran, Ifrad) determine the order of "
            "Umrah/Hajj and whether a sacrifice is required; the pilgrim chooses one at Ihram."
        ),
    },
    {
        "id": "official_apps",
        "keywords_ar": ["تطبيق نسك", "التطبيقات الرسمية", "نسك"],
        "keywords_en": ["nusuk", "official app", "official apps"],
        "title_ar": "التطبيقات والمنصات الرسمية",
        "title_en": "Official Apps and Platforms",
        "body_ar": (
            "- منصة وتطبيق نسك (Nusuk) هي المنصة الرسمية لحجز تصاريح الحج والعمرة وزيارة "
            "الحرمين، وتقدم معلومات إرشادية موثوقة.\n"
            "- تصدر وزارة الحج والعمرة والهيئة العامة للعناية بشؤون المسجدين تحديثات وتطبيقات "
            "رسمية إضافية بحسب الموسم — يُنصح دائمًا بالتحقق من القنوات الرسمية لأحدث المعلومات."
        ),
        "body_en": (
            "- The Nusuk platform/app is the official channel for booking Hajj and Umrah permits and "
            "visits to the two mosques, and offers reliable guidance.\n"
            "- The Ministry of Hajj and Umrah and the General Presidency for the Affairs of the Two "
            "Holy Mosques issue additional official updates and apps each season — always check "
            "official channels for the latest information."
        ),
    },
]


# Shared with ai_pipeline.py (relevance filter) and assistant.py (scope
# check) so the "is this about Hajj/Umrah?" definition lives in ONE place.
# Mirrors the topic list in the product spec: rituals, Ihram, Tawaf, Sa'i,
# Arafah, Muzdalifah, Mina, Jamarat, Hady, Halq/Taqsir, Miqats, Ihram
# prohibitions, Fidyah, duas/adhkar, shar'i rulings, and every listed
# Haramain service (fatwa/guidance offices, study areas, Qur'an circles,
# restrooms, ablution areas, gates, prayer areas, elderly/disability carts,
# first aid, health centers, lost & found, crowd management, transport,
# official apps) plus the broader Hajj/Umrah travel & services vocabulary
# used to judge external reviews (hotels, transport companies, campaigns).
TOPIC_WORDS = {
    "hajj", "umrah", "mecca", "makkah", "madinah", "medina", "kaaba", "tawaf",
    "ihram", "haram", "pilgrimage", "pilgrim", "zamzam", "arafat", "arafah", "mina",
    "muzdalifah", "safa", "marwa", "marwah", "jamarat", "rawdah", "masjid",
    "miqat", "meeqat", "talbiyah", "fidyah", "hady", "sacrifice", "sai",
    "nusuk", "grand mosque", "prophet's mosque", "prophets mosque",
    "حج", "الحج", "عمرة", "العمرة", "مكة", "المدينة", "الحرم", "الكعبة",
    "طواف", "احرام", "الإحرام", "زمزم", "عرفات", "عرفة", "منى", "مزدلفة", "الصفا",
    "المروة", "الجمرات", "الروضة", "المسجد", "معتمر", "حاج", "الحجاج", "المعتمرين",
    "المشاعر المقدسة", "التفويج", "المطاف", "المسعى", "ميقات", "المواقيت",
    "محظورات الإحرام", "فدية", "الهدي", "الحلق", "التقصير", "طواف الإفاضة",
    "طواف الوداع", "الأدعية", "الأذكار", "الأحكام الشرعية", "نسك",
    "المسجد الحرام", "المسجد النبوي", "مكاتب الإفتاء", "مكاتب الإرشاد",
    "حلقات القرآن", "دورات المياه", "أماكن الوضوء", "البوابات", "المصليات",
    "عربات كبار السن", "عربات ذوي الإعاقة", "الإسعافات الأولية",
    "المراكز الصحية", "المفقودات", "تنظيم الحشود", "خدمات التنقل",
    "حملة حج", "شركة حج", "شركة عمرة", "فندق حجاج", "نقل حجاج",
}


def _norm(s: str) -> str:
    s = (s or "").lower()
    # Normalize common Arabic letter variants so "أركان"/"اركان", "إحرام"/
    # "احرام", "الميقات"/"الميقآت" etc. all match the same lexicon entry —
    # pilgrims frequently type without hamzas on a phone keyboard.
    s = re.sub(r"[إأآا]", "ا", s)
    s = s.replace("ة", "ه").replace("ى", "ي")
    s = re.sub(r"\s+", " ", s).strip()
    return s


_WORD_RE = re.compile(r"[\w\u0600-\u06FF']+")


def _words(s: str) -> set:
    return set(_WORD_RE.findall(_norm(s)))


def retrieve(query: str, limit: int = 3):
    """Small keyword-overlap retriever — no embeddings, no network calls.

    Two passes, both on the normalized text (see _norm — handles hamza/taa
    marbuta variants and casing):
      1. Substring pass: full multi-word keyword phrases found anywhere in
         the query (strongest signal, e.g. "الحجر الأسود").
      2. Word-overlap pass: any keyword that shares at least one whole word
         with the query (catches partial/reordered phrasing, e.g. a query
         about "اركان العمره" still hits an entry keyed on "العمرة").
    A hit from pass 1 counts double so precise phrase matches still rank
    above loose single-word overlaps."""
    q_norm = _norm(query)
    if not q_norm:
        return []
    q_words = _words(query)
    scored = []
    for entry in KB:
        score = 0
        for kw in entry["keywords_ar"] + entry["keywords_en"]:
            if not kw:
                continue
            kw_norm = _norm(kw)
            if kw_norm in q_norm:
                score += 2
            elif _words(kw) & q_words:
                score += 1
        if score:
            scored.append((score, entry))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [e for _, e in scored[:limit]]


def in_scope(text: str) -> bool:
    """Is this text about Hajj/Umrah/Haramain services? Same normalization
    as retrieve() (hamza/taa-marbuta variants, casing) so spelling slips
    don't cause a false "out of scope" reply. Shared by assistant.py's
    scope guard and ai_pipeline.py's external-comment relevance filter —
    one definition, used everywhere "is this on-topic?" is asked."""
    hay = _norm(text)
    if not hay:
        return False
    hay_words = _words(text)
    for kw in TOPIC_WORDS:
        kw_norm = _norm(kw)
        if kw_norm and kw_norm in hay:
            return True
        if _words(kw) & hay_words:
            return True
    return False


def context_block(query: str, lang: str = "ar", limit: int = 3) -> str:
    """Formatted grounding block to inject into the assistant's system
    prompt — empty string if nothing in the KB matches the query."""
    hits = retrieve(query, limit=limit)
    if not hits:
        return ""
    use_ar = lang not in ("en",)
    lines = []
    for e in hits:
        if use_ar:
            lines.append(f"## {e['title_ar']}\n{e['body_ar']}")
        else:
            lines.append(f"## {e['title_en']}\n{e['body_en']}")
    return "\n\n".join(lines)
