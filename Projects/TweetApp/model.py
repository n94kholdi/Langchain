import os
from langchain_cohere import ChatCohere
from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate


class TweetGeneratorModel:
    def __init__(self, api_key, language='fa'):
        os.environ["COHERE_API_KEY"] = api_key
        self.language = language
        self.translations = self.get_translations()
        self.chat_model = ChatCohere(streaming=True, temperature=0.7)
        self.examples = self.get_examples()
        self.few_shot_prompt = self.create_few_shot_prompt()

    def get_translations(self):
        return {
            'fa': {
                'system_prompt': '''یک مدل که با توجه به ژانر ورودی یک توییت تحویل می‌دهد،
                توجه داشته باش کاربران این توییت ایرانی هستند و با توجه به فرهنگ این کشور پاسخ بده'''
            },
            'en': {
                'system_prompt': '''A model that generates a tweet based on the input genre.
                Keep in mind that the users of this tweet are Iranian, so respond according to their culture.'''
            }
        }

    def get_examples(self):
        return [
            {"language":"fa","genre": "طنز تلخ", "tweet": "ایران، کشوریه که اگر به عشق و آرزوهای آدماش نگاه کنید، می‌تونید یه فیلم درام بسازید. اگر بخواید به واقعیت نگاه کنید، یه فیلم کمدی!😌"},
            {"language":"fa","genre": "طنز تلخ", "tweet": "در ایران، وقتی می‌گی«دارم تلاش می‌کنم»، همه فکر می‌کنند داری دنبال یک کار خوب می‌گردی. واقعیت این است که داری دنبال یک اینترنت پرسرعت می‌گردی."},
            {"language":"fa","genre": "اجتماعی", "tweet": "ایران، جایی که هر کسی یک داستان دارد و همه می‌دانند که داستان‌هایشان با «یک روز» شروع می‌شود و با «خدا رو شکر» تمام می‌شود"},
            {"language":"fa","genre": "اجتماعی", "tweet": "در عصر دیجیتال، شاید راحت‌تر از همیشه بتوانیم نظرات خود را به اشتراک بگذاریم، اما هنوز هم مهم است که یاد بگیریم چگونه با احترام به نظرات دیگران گوش دهیم و گفت‌وگو کنیم."},
            {"language":"fa","genre": "طنز تلخ", "tweet": "برنامه‌ی اصلی من برای امروز: ۱- بیدار شدن. ۲-بیدار باقی‌موندن. ۳- تلاش برای این‌که دوباره نخوابم."},
            {"language":"fa","genre": "جوک", "tweet": "حالا گاوداری و مرغداری و پرورش اسب و این چیزا رو خارج شهر می‌سازن قابل درکه برام، اما نمی‌فهمم دیگه چرا دانشگاه‌ها رو خارج شهر می‌سازن؟"},
            {"language":"fa","genre": "جوک", "tweet": "وقتي داري به آينده ات فکر ميکني، \nيکمم به اوني که پشت در دستشويى داره زمينو چنگ ميزنه فکر کن"},
            {"language":"fa","genre": "علمی", "tweet": "در فضا، به دلیل نبود جاذبه، مایعات به شکل کروی درمی‌آیند. این بدان معناست که اگر یک قطره آب در فضا بریزید، به صورت توپ‌های کوچک آب معلق خواهد بود!"},
            {"language":"fa","genre": "علمی", "tweet": "مغز انسان تنها ۲٪ از وزن کل بدن را تشکیل می‌دهد، اما به طور مداوم حدود ۲۰٪ از انرژی بدن را مصرف می‌کند!"},
            {"language":"fa","genre": "علمی", "tweet": "مغز ما به طور پیوسته در حال به‌روزرسانی حافظه‌هاست. به همین دلیل است که بعضی خاطرات قدیمی با گذشت زمان تغییر می‌کنند."},
            {"language":"fa","genre": "علمی", "tweet": "اگر DNA موجود در سلول‌های بدن ما رو باز کنیم، طول اون به حدود ۲ متر می‌رسه! جالب نیست که این‌همه اطلاعات درون یک سلول کوچک جا می‌شن؟"},
            {"language":"fa","genre": "علمی", "tweet": "قلب انسان در طول زندگی حدود ۲.۵ میلیارد بار می‌تپد. این یعنی قلب شما یکی از سخت‌کوش‌ترین عضلات بدنتونه!"},
            {"language":"fa","genre": "طنز تلخ", "tweet": "فکر می‌کنی که دیگه همه چیو تو زندگی امتحان کردی، یه بار امتحان کن با اینترنت ایران فیلم آنلاین ببینی!"},
            {"language":"fa","genre": "طنز تلخ", "tweet": "چالش زندگی من: «چرا همیشه وقتی می‌خوام ورزش کنم، حس خوابم بیشتره؟!»"},
            {"language":"fa","genre": "جوک", "tweet": "ﮐـﯽ ﮔﻔـﺘﻪ ﺩﯾـﻔﺮﺍﻧﺴـﻞ ﻭ ﺍﻧﺘـﮕﺮﺍﻝ ﻫﯿـﭻ ﺟـﺎﯼﺯﻧـﺪﮔﯽ ﺑـﻪ ﺩﺭﺩ نمی خوره ؟! هان؟ ﺑﻨـﺪﻩ ﺧـﻮﺩﻡ ﺑـﻪ ﺷـﺨـﺼﻪ ﺩﯾـﺮﻭﺯ ﺗﻤـﺎﻡ ﺷﯿـﺸـﻪ ﻫﺎیﺧـﻮﻧـﻤﻮﻧﻮ ، ﺑﺎ ﺟـﺰﻭﻩ های همیـﻨﺎ ﭘـﺎﮎ ﮐـﺮﺩﻡ"},
            {"language":"fa","genre": "جوک", "tweet": "ﻣﻦ ﻣﻮﻧﺪﻡ ﮐﺮﻭﮐﺪﯾﻞ ﺧﺠﺎﻟﺖ ﻧﻤﯿﮑﺸﻪ ﺑﺎ ﺍﻭﻥ ﻫﯿﮑﻠﺶ ﺗﺨﻢ ﻣﯿﺬﺍﺭﻩ؟"},
            {"language": "en", "genre": "Dark Humor", "tweet": "Life in the modern world: You either die a hero, or live long enough to see yourself replying to work emails at 2 AM."},
            {"language": "en", "genre": "Dark Humor", "tweet": "They say laughter is the best medicine, but it doesn't seem to work when you're choking on your student loans."},
            {"language": "en", "genre": "Dark Humor", "tweet": "Growing up, I was told I could be anything I wanted. Turns out, what I wanted most was a nap."},
            {"language": "en", "genre": "Joke", "tweet": "Why don't skeletons fight each other? They don't have the guts!"},
            {"language": "en", "genre": "Joke", "tweet": "I told my computer I needed a break. Now it won't stop sending me vacation ads."},
            {"language": "en", "genre": "Joke", "tweet": "Why did the scarecrow win an award? Because he was outstanding in his field."},
            {"language": "en", "genre": "Science", "tweet": "In space, there's no sound. So if you scream because you forgot your homework, no one will hear you."},
            {"language": "en", "genre": "Science", "tweet": "Your brain is about 75% water, so technically, you’re constantly thinking about hydration."},
            {"language": "en", "genre": "Science", "tweet": "The human body has over 37 trillion cells. At least one of them should be able to remember where I put my keys."},    
            {"language": "en", "genre": "Social", "tweet": "In today's world, we are more connected than ever, yet somehow, we’re still figuring out how to communicate."},
            {"language": "en", "genre": "Social", "tweet": "Social media: where everyone's highlight reel is someone else's measure of success."},
            {"language": "en", "genre": "Social", "tweet": "Life isn’t just about milestones; it’s about the little moments in between that nobody posts about."}
        ]

    def create_few_shot_prompt(self):
        example_prompt = ChatPromptTemplate.from_messages(
            [("human", "{genre}"), ("ai", "{tweet}")]
        )

        return FewShotChatMessagePromptTemplate(
            example_prompt=example_prompt,
            examples=self.examples,
        )

    def generate_tweet(self, genre):
        final_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", self.translations[self.language]['system_prompt']),
                self.few_shot_prompt,
                ("human", "{genre}")
            ]
        )
        chain = final_prompt | self.chat_model
        return chain.invoke({"language": self.language, "genre": genre}).content
