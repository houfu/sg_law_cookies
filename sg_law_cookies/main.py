import datetime

import requests
from bs4 import BeautifulSoup
from jinja2 import Environment, PackageLoader
from langchain.chat_models import ChatOpenAI
from langchain.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate, ChatPromptTemplate


class NewsArticle:
    def __init__(self, article):
        self.category = article.category.text
        self.title = article.title.text
        self.source_link = article.link.text
        self.author = article.author.text
        self.date = datetime.datetime.strptime(article.pubDate.text, "%d %b %Y %H:%M:%S")


def check_if_article_should_be_included(article: NewsArticle, scrape_date: datetime.date) -> bool:
    """
    Checks if an article should be included in the list to be processed by:

    * Removing articles which are advertisements
    * Removing articles which are in the past (i.e. not today). If it is a weekend today, return Friday's articles.

    :param scrape_date:
    :param article:
    :return:
    """
    if article.category == "Singapore Law Watch":
        return False
    if article.title.startswith('ADV: '):
        return False
    today = scrape_date
    if today.weekday() == 0:
        saturday = today - datetime.timedelta(days=2)
        return article.date.year == saturday.year and article.date.day >= saturday.day \
            and article.date.month == saturday.month
    else:
        return article.date.year == today.year and article.date.day == today.day and article.date.month == today.month


def scrape_news_articles_today(scrape_date: datetime.date) -> list[NewsArticle]:
    """
    Returns a list of today's news articles from Singapore law watch.
    :return:
    """
    rss_link = "https://www.singaporelawwatch.sg/Portals/0/RSS/Headlines.xml"

    r = requests.get(rss_link)

    soup = BeautifulSoup(r.content, 'lxml-xml')

    news_articles = [NewsArticle(article) for article in soup.find_all('item')]

    return [article for article in news_articles if
            check_if_article_should_be_included(article, scrape_date)]


def get_summaries(articles: list[NewsArticle]):
    chat = ChatOpenAI(temperature=0.25)
    system_template = "You are a helpful assistant that summarises news articles and opinions for lawyers who " \
                      "are in a rush. \nThe summary should focus on the legal aspects of the article and be " \
                      "accessible and easygoing. \n Summaries should be accurate and engaging."
    system_message_prompt = SystemMessagePromptTemplate.from_template(system_template)
    human_template = "Summarise this article from the Singapore Law Watch website: \n\n {article}"
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)
    shorter_template = "Now make it more concise."
    shorter_message_prompt = HumanMessagePromptTemplate.from_template(shorter_template)
    article_summary_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])

    day_messages = [] + system_message_prompt.format_messages()
    summaries = []

    for article in articles:
        r = requests.get(article.source_link)
        soup = BeautifulSoup(r.content, "html5lib")
        article_content = soup.article.h1.text + "\n" + "\n".join([p.text for p in soup.article.find_all('p')])

        messages = article_summary_prompt.format_prompt(article=article_content).to_messages()

        result = chat(messages)

        while len(result.content) > 450:
            messages.append(result)
            messages = messages + shorter_message_prompt.format_messages()
            result = chat(messages)

        summaries.append((result.content, article.source_link))

        day_messages.append(result)

    day_summary_template = "Now, use all the news articles provided previously to create an introduction " \
                           "for today's blog post in the form of a vivid poem which has not more than 6 lines."
    day_summary_prompt = HumanMessagePromptTemplate.from_template(day_summary_template)

    day_messages = day_messages + day_summary_prompt.format_messages()

    day_summary = chat(day_messages)

    return summaries, day_summary.content.splitlines()


def main():
    print("Let's start.")
    env = Environment(
        loader=PackageLoader("sg_law_cookies")
    )
    template = env.get_template('template.jinja2')
    print("Getting summaries.")
    scrape_date = datetime.datetime.today()
    summaries, day_summary = get_summaries(scrape_news_articles_today(scrape_date))
    if len(summaries) == 0:
        raise Exception("No summaries were found.")
    print("Summaries completed, rendering template.")
    print(template.render(
        today=scrape_date.strftime("%d %B %Y"),
        summaries=summaries,
        day_summary=day_summary
    ))

    blog_template = env.get_template("blog_post.jinja2")
    with open(f'site/content/post/{scrape_date.strftime("%d-%B-%Y")}.md',
              mode='x') as file:
        file.write(
            blog_template.render(
                today=scrape_date,
                summaries=summaries,
                day_summary_block='  \n  '.join(day_summary),
                day_summary='  \n'.join(day_summary)
            )
        )

    newsletter_template = env.get_template("newsletter_post_html.jinja2")
    content_html = newsletter_template.render(
        today=scrape_date,
        summaries=summaries,
        day_summary=day_summary,
    )
    newsletter_template_text = env.get_template("newsletter_post_text.jinja2")
    content = newsletter_template_text.render(
        today=scrape_date,
        summaries=summaries,
        day_summary='\n'.join(day_summary)
    )
    title = f"SG Law Cookies ({scrape_date.strftime('%d %B %Y')})"
    response_email = requests.post(
        "https://cookies.your-amicus.app/sg-law-cookies-func/email_support/send_newsletter",
        json={"content_html": content_html, "context_text": content, "title": title},
        headers={"Content-Type": "application/json"}
    )
    print(f"Email newsletter: {response_email.json()['message']}")


if __name__ == '__main__':
    main()
