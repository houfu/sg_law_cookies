import datetime

import requests
from bs4 import BeautifulSoup
from jinja2 import Environment, PackageLoader
from mirascope import BaseMessageParam, llm, prompt_template
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

system_message_prompt = BaseMessageParam(role="system", content=system_template)


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

    result = [
        article
        for article in news_articles
        if check_if_article_should_be_included(article, scrape_date)
    ]
    print(f"No of articles: {len(result)}")
    return result


@llm.call(
    provider="openai", model="gpt-4.1-mini", temperature=0.4, response_model=NewsArticle
)
@prompt_template(
    """
    SYSTEM: 
    As an AI expert in legal affairs, your task is to provide concise, yet comprehensive summaries of legal news 
    articles for time-constrained attorneys. These summaries should highlight the critical legal aspects, 
    relevant precedents, and implications of the issues discussed in the articles.
    
    Despite their complexity, the summaries should be accessible and digestible, written in an engaging and
    conversational style. Accuracy and attention to detail are essential, as the readers will be legal professionals who 
    may use these summaries to inform their practice.
    
    ### Instructions: 
    1. Begin the summary with a brief introduction of the topic of the article.
    2. Outline the main legal aspects, implications, and precedents highlighted in the article. 
    3. End the summary with a succinct conclusion or takeaway.
    
    The summaries should not be longer than 100 words, but ensure they efficiently deliver the key legal insights,
    making them beneficial for quick comprehension. The end goal is to help the lawyers understand the crux of the 
    articles without having to read them in their entirety
    
    USER:
    Article from Singapore Law Watch ({article_date}):
    Title: {article.title}
    Category: {article.category}

    Content:
    {article_content}

    Provide a structured legal analysis following the format above.
    """
)
def get_summary(article: ScrapedArticle) -> NewsArticle:
    r = requests.get(article.source_link)
    soup = BeautifulSoup(r.content, "html5lib")
    article_content = (
        soup.article.h1.text
        + "\n"
        + "\n".join([p.text for p in soup.article.find_all("p")])
    )
    article_date = article.date.strftime("%d %B %Y")

    return {
        "computed_fields": {
            "article_date": article_date,
            "article_content": article_content,
        }
    }


@llm.call(provider="openai", model="gpt-4.1", temperature=0.8)
@prompt_template(
    """
    SYSTEM:
    As an expert poet, your challenge is to craft a succinct yet vivid poem of no more than six lines.
    Your poem should encapsulate the essence of the news summaries provided below.

    USER:
    {text}
    """
)
def get_day_summary(text: str): ...


def get_summaries(articles: list[ScrapedArticle]):

    day_messages = [] + system_message_prompt.format_messages()
    summaries = []

    for article in articles:
        result = get_summary(article)

        summaries.append((result.summary, article.source_link))

        day_messages.append(f"Here is a summary: \n\n {result.summary}\n\n")

        # requests.post(
        #     "https://cookies.zeeker.sg/sg-law-cookies-func/zeeker_support/new_newsarticle",
        #     json={"content": result.model_dump_json()},
        #     headers={"Content-Type": "application/json"},
        # )

    day_summary: llm.CallResponse = get_day_summary("\n".join(day_messages))

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
    with open(f'content/post/{scrape_date.strftime("%d-%B-%Y")}.md', mode="x") as file:
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
