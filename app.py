"""Streamlit Playground"""

import os
import time

import streamlit as st

IMAGE_FILENAME = "trial chamber.png"


def main():
    st.set_page_config("Streamlit Playground", layout="wide")
    st.html("""
                <style>
                    pre {
                        max-height: 500px;
                        overflow: auto;
                    }
                </style>
            """)

    left, center, right = st.columns((2, 3, 1))

    with left:
        image_download()

    with center:
        self_download()

    with right:
        st.header("Slow Task Example")
        slow_task()


def slow_task():
    st.write("experimenting with `st.spinner`")

    with st.spinner(), st.empty():
        for i in range(10):
            st.write(f"{i}...")
            time.sleep(1)
        else:
            st.write("done!")
        st.button("Again!")


def image_download():
    with open(IMAGE_FILENAME, "rb") as fin:
        image_contents = fin.read()

    image_name = os.path.basename(IMAGE_FILENAME)

    st.subheader(image_name)
    st.image(image_contents)
    st.download_button("Download Image", image_contents, file_name=image_name)


def self_download():
    with open(__file__, encoding="utf-8") as fin:
        my_contents = fin.read()

    my_name = os.path.basename(__file__)

    st.subheader(my_name)
    st.code(my_contents)
    st.download_button("Download Source", my_contents, file_name=my_name)


main()
