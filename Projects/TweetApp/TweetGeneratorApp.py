import streamlit as st
from model import TweetGeneratorModel 

class TweetGeneratorApp:
    def __init__(self):
        self.translations = {
            'fa': {
                'change_language': 'تغییر زبان',
                'current_language': 'زبان فعلی: فارسی',
                'api_key_input': 'کلید API خود را وارد کنید.',
                'app_title': '🚀 توییت ساز',
                'genre_label': 'یک ژانر را انتخاب کنید.',
                'tweet_count': 'تعداد توییت‌ها',
                'generate_button': 'توییت بساز',
                'placeholder_genre': 'انتخاب ژانر',
                'genres': ['علمی', 'جوک', 'اجتماعی', 'طنز تلخ'],
                'help':'یک عدد بین ۰ تا ۵ را انتخاب کنید'
            },
            'en': {
                'change_language': 'Change Language',
                'current_language': 'Current Language: English',
                'api_key_input': 'Enter your API key.',
                'app_title': '🚀 Tweet Generator',
                'genre_label': 'Select a genre.',
                'tweet_count': 'Number of Tweets',
                'generate_button': 'Generate Tweet',
                'placeholder_genre': 'Select Genre',
                'genres': ['Science', 'Joke', 'Social', 'Dark Humor'],
                'help': 'Select a Number between 1 to 5'
            }
        }

        if 'language' not in st.session_state:
            st.session_state.language = 'fa'

        if st.button(self.translations[st.session_state.language]['change_language']):
            st.session_state.language = 'en' if st.session_state.language == 'fa' else 'fa'

        self.language = st.session_state.language
        self.text_align, self.direction = self.get_text_alignment()
        self.inject_css()
        self.model = None

    def get_text_alignment(self):
        if self.language == 'fa':
            return 'right', 'rtl'
        else:
            return 'left', 'ltr'

    def inject_css(self):
        st.markdown(f"""
            <style>
                    
                @import url('https://fonts.googleapis.com/css2?family=Vazirmatn:wght@400;500&display=swap');
                 
                * {{
                    font-family: 'Vazirmatn', sans-serif;
                    direction: {self.direction};
                    text-align: {self.text_align};
                }}

            </style>

        """, unsafe_allow_html=True)

    def display_app(self):

        st.write(self.translations[self.language]['current_language'])
        api_key = st.text_input(self.translations[self.language]['api_key_input'], type="password")

        if not api_key:
            st.stop()
            
        st.header(self.translations[self.language]['app_title'], divider='orange') 

        self.model = TweetGeneratorModel(api_key, self.language)

        form = st.form(key="user_settings")

        with form:
            genres = self.translations[self.language]['genres']
            user_genere = st.selectbox(self.translations[self.language]['genre_label'], genres, 
                                       index=None, placeholder=self.translations[self.language]['placeholder_genre'])

            num_input = st.slider(self.translations[self.language]['tweet_count'], value=1, key="num_input", min_value=1, max_value=5,
                                  help=self.translations[self.language]['help'])

            generate_button = form.form_submit_button(self.translations[self.language]['generate_button'])

        if generate_button and user_genere:
            for _ in range(num_input):
                st.divider()
                tweet = self.model.generate_tweet(user_genere)
                st.write(tweet)

if __name__ == "__main__":
    app = TweetGeneratorApp()
    app.display_app()