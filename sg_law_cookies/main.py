import datetime

import requests
from bs4 import BeautifulSoup
from jinja2 import Environment, PackageLoader
from langchain.prompts import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    ChatPromptTemplate,
    AIMessagePromptTemplate,
)
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, AnyHttpUrl

system_template = """As an AI expert in legal affairs, your task is to provide concise, yet comprehensive 
    summaries of legal news articles for time-constrained attorneys. These summaries should highlight the critical 
    legal aspects, relevant precedents, and implications of the issues discussed in the articles.

Despite their complexity, the summaries should be accessible and digestible, written in an engaging and 
conversational style. Accuracy and attention to detail are essential, as the readers will be legal professionals who 
may use these summaries to inform their practice.

### Instructions: 
1. Begin the summary with a brief introduction of the topic of the article.
2. Outline the main legal aspects, implications, and precedents highlighted in the article. 
3. End the summary with a succinct conclusion or takeaway.

The summaries should not be longer than 100 words, but ensure they efficiently deliver the key legal insights, 
making them beneficial for quick comprehension. The end goal is to help the lawyers understand the crux of the 
articles without having to read them in their entirety."""

system_message_prompt = SystemMessagePromptTemplate.from_template(system_template)


class NewsArticle(BaseModel):
    category: str
    title: str
    source_link: AnyHttpUrl
    author: str
    date: datetime.date
    summary: str
    text: str


class ScrapedArticle:
    category: str
    title: str
    source_link: AnyHttpUrl
    author: str
    date: datetime.date

    @classmethod
    def from_article(cls, article):
        self = cls()
        self.category = article.category.text
        self.title = article.title.text
        self.source_link = article.link.text
        self.author = article.author.text
        self.date = datetime.datetime.strptime(
            article.pubDate.text, "%d %b %Y %H:%M:%S"
        ).date()
        return self


class SGLawCookie(BaseModel):
    resource_url: AnyHttpUrl
    cookie_content: str
    published_date: datetime.date


def check_if_article_should_be_included(
    article: ScrapedArticle, scrape_date: datetime.date
) -> bool:
    """
    Checks if an article should be included in the list to be processed by:

    * Removing articles which are advertisements
    * Removing articles which are in the past (i.e. not today).
    If it is a Monday today, return Saturday and Sunday's articles.

    :param scrape_date:
    :param article:
    :return:
    """
    if article.category == "Singapore Law Watch":
        return False
    if article.title.startswith("ADV: "):
        return False
    today = scrape_date
    date_filter = (
        [
            today - datetime.timedelta(days=2),  # Saturday
            today - datetime.timedelta(days=1),  # Sunday
            today,  # Monday
        ]
        if today.weekday() == 0
        else [today]
    )
    return article.date in date_filter


def scrape_news_articles_today(scrape_date: datetime.date) -> list[ScrapedArticle]:
    """
    Returns a list of today's news articles from Singapore law watch.
    :return:
    """
    rss_link = "https://www.singaporelawwatch.sg/Portals/0/RSS/Headlines.xml"

    r = requests.get(rss_link)

    soup = BeautifulSoup(r.content, "lxml-xml")

    news_articles = [
        ScrapedArticle.from_article(article) for article in soup.find_all("item")
    ]

    result = [article for article in news_articles if check_if_article_should_be_included(article, scrape_date)]
    print(f"No of articles: {len(result)}")
    return result


def get_summary(article: ScrapedArticle) -> NewsArticle:
    r = requests.get(article.source_link)
    soup = BeautifulSoup(r.content, "html5lib")
    article_content = (
        soup.article.h1.text
        + "\n"
        + "\n".join([p.text for p in soup.article.find_all("p")])
    )

    human_template = (
        """
        Article from Singapore Law Watch ({date}):
        Title: {title}
        Category: {category}

        Content:
        {article}

        Provide a structured legal analysis following the format above.
        """
    )
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)
    article_summary_prompt = ChatPromptTemplate.from_messages(
        [system_message_prompt, human_message_prompt]
    )
    messages = article_summary_prompt.format_prompt(
        article=article_content,
        date=article.date.strftime("%d %B %Y"),
        title=article.title,
        category=article.category,
    ).to_messages()
    chat = ChatOpenAI(model="gpt-4o-mini", temperature=0.4)
    summary_response = chat.invoke(messages)
    return NewsArticle(
        category=article.category,
        title=article.title,
        source_link=article.source_link,
        author=article.author,
        date=article.date,
        summary=summary_response.content,
        text=article_content,
    )


def get_summaries(articles: list[ScrapedArticle]):
    llm_template = "Here is a summary: \n\n {summary}"
    llm_message_prompt = AIMessagePromptTemplate.from_template(llm_template)

    day_messages = [] + system_message_prompt.format_messages()
    summaries = []

    for article in articles:
        result = get_summary(article)

        summaries.append((result.summary, article.source_link))

        day_messages.append(llm_message_prompt.format(summary=result.summary))

        # requests.post(
        #     "https://cookies.zeeker.sg/sg-law-cookies-func/zeeker_support/new_newsarticle",
        #     json={"content": result.model_dump_json()},
        #     headers={"Content-Type": "application/json"},
        # )

    day_summary_template = """As an expert poet, your challenge is to craft a succinct yet vivid poem of no more than 
    six lines. This poem should encapsulate the essence of the multiple news summaries previously provided.

### Your Toolkit:
- Start with clear instructions.
- Make use of descriptive language and powerful imagery to keep your reader engaged.
- Experiment with various poetic techniques such as alliteration, rhyme or metaphor.
- Your primary goal is to create a snapshot of the current world scenario through your verse.

Example:

"In the spins of world affairs, where facts unfurl.<br> 
Through winds of change, the news summary swirls..."
        """
    day_summary_prompt = HumanMessagePromptTemplate.from_template(day_summary_template)

    day_messages = day_messages + day_summary_prompt.format_messages()

    chat = ChatOpenAI(model="gpt-4o", temperature=0.1)

    day_summary = chat.invoke(day_messages)

    return summaries, day_summary.content.splitlines()


def main():
    print("Let's start.")
    env = Environment(loader=PackageLoader("sg_law_cookies"))
    template = env.get_template("template.jinja2")
    print("Getting summaries.")
    scrape_date = datetime.date.today()
    summaries, day_summary = get_summaries(scrape_news_articles_today(scrape_date))
    if len(summaries) == 0:
        raise Exception("No summaries were found.")
    print("Summaries completed, rendering template.")
    content = template.render(
        today=scrape_date.strftime("%d %B %Y"),
        summaries=summaries,
        day_summary=day_summary,
    )
    print(content)

    blog_template = env.get_template("blog_post.jinja2")
    with open(
        f'site/content/post/{scrape_date.strftime("%d-%B-%Y")}.md', mode="x"
    ) as file:
        file.write(
            blog_template.render(
                today=scrape_date,
                summaries=summaries,
                day_summary_block="  \n  ".join(day_summary),
                day_summary="  \n".join(day_summary),
            )
        )

    new_cookie = SGLawCookie(
        resource_url=f"https://cookies.zeeker.sg/post/{scrape_date.strftime('%d-%B-%Y').lower()}/",
        cookie_content=content,
        published_date=datetime.date.today(),
    )

    requests.post(
        "https://cookies.zeeker.sg/sg-law-cookies-func/zeeker_support/new_cookie",
        json={"content": new_cookie.model_dump_json()},
        headers={"Content-Type": "application/json"},
    )

    newsletter_template = env.get_template("newsletter_post_html.jinja2")
    content_html = newsletter_template.render(
        today=scrape_date,
        summaries=summaries,
        day_summary=day_summary,
    )
    newsletter_template_text = env.get_template("newsletter_post_text.jinja2")
    content = newsletter_template_text.render(
        today=scrape_date, summaries=summaries, day_summary="\n".join(day_summary)
    )
    title = f"SG Law Cookies ({scrape_date.strftime('%d %B %Y')})"
    response_email = requests.post(
        "https://cookies.zeeker.sg/sg-law-cookies-func/email_support/send_newsletter",
        json={"content_html": content_html, "context_text": content, "title": title},
        headers={"Content-Type": "application/json"},
    )
    print(f"Email newsletter: {response_email.json()['message']}")


if __name__ == "__main__":
    main()
