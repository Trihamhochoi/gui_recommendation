import streamlit as st
import os
import re
import base64
from pathlib import Path
import math
import locale


def load_bootstrap():
    return st.markdown(
        '<link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0/css/bootstrap.min.css" integrity="sha384-Gn5384xqQ1aoWXA+058RXPxPg6fy4IWvTNh0E263XmFcJlSAwiGgFAW/dAiS6JXm" crossorigin="anonymous">',
        unsafe_allow_html=True)


def get_dict_word(file_path: str) -> dict:
    with open(file_path, 'r', encoding="utf8") as file:
        _lst = file.read().split('\n')
        _dict = {}
        for line in _lst:
            key, value = line.split('\t')
            _dict[key] = str(value)

    return _dict


def markdown_images(markdown):
    # example image markdown:
    # ![Test image](images/test.png "Alternate text")
    images = re.findall(r'(!\[(?P<image_title>[^\]]+)\]\((?P<image_path>[^\)"\s]+)\s*([^\)]*)\))', markdown)
    return images


def img_to_bytes(img_path):
    img_bytes = Path(img_path).read_bytes()
    encoded = base64.b64encode(img_bytes).decode()
    return encoded


def img_to_html(img_path, img_alt):
    img_format = img_path.split(".")[-1]
    img_html = f'<img src="data:image/{img_format.lower()};base64,{img_to_bytes(img_path)}" alt="{img_alt}" style="max-width: 100%;">'

    return img_html


def markdown_insert_images(markdown):
    images = markdown_images(markdown)

    for image in images:
        image_markdown = image[0]
        image_alt = image[1]
        image_path = image[2]
        if os.path.exists(image_path):
            markdown = markdown.replace(image_markdown, img_to_html(image_path, image_alt))
    return markdown


# def format_vietnamese_currency(number):
#     # Set the locale to use Vietnamese (Vietnam)
#     locale.setlocale(locale.LC_ALL, 'vi_VN.UTF-8')
#
#     # Format the number as currency (Vietnamese đồng)
#     formatted_currency = locale.currency(number, grouping=True)
#
#     return formatted_currency

def format_vnd(number):
    """
    Format a number as Vietnamese đồng (VND) currency.
    """
    # Add thousand separators
    formatted_number = "{:,.0f}".format(number)
    # Add đồng symbol
    formatted_number += " ₫"
    return formatted_number


def view_info_prd(list_prd: list):
    for info_dict in list_prd:
        start_ = '⭐' * math.floor(info_dict['rating'])
        st.write(f'### {info_dict["product_name"]}')
        st.write(f"""
        - **ProductID:** {info_dict['product_id']}
        - **Sub Category:** {info_dict['sub_category']}
        - **Price:** {format_vnd(info_dict['price'])}
        - **Rating:** {start_}
        - **Product Link:** [Click here]({info_dict['link']})
        """)

        if isinstance(info_dict['image'], str):
            st.image(info_dict['image'],
                     caption=info_dict['product_name'],
                     use_column_width=True)
        # url = info_dict['image']
        # response = requests.get(url)
        # img = Image.open(BytesIO(response.content))


if __name__ == '__main__':
    md_path = '../DATA/report_md/recommendation_sysyten_report.md'
    with open(md_path, 'r', encoding='utf-8') as file:
        markdown_text = file.read()

    readme_img = markdown_insert_images(markdown_text)
    print(readme_img)
