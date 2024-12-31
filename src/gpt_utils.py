import argparse

import requests
import json

url = "https://api.openai-hk.com/v1/chat/completions"

headers = {
    "Content-Type": "application/json",
    "Authorization": "Bearer hk-v03dbg100002436825fa3ceaa2c69016ca14833e0d837694"
}


def get_anchor_concepts(ablated_concept, num):
    data = {
        "max_tokens": 1200,
        "model": "gpt-3.5-turbo",
        "temperature": 0.5,
        "top_p": 1,
        "presence_penalty": 1,
        "messages": [
            {
                "role": "system",
                "content": "You are ChatGPT, a large language model trained by OpenAI. Answer as concisely as possible."
            },
            {
                "role": "user",
                "content": "find 20  word or phrase of concepts about Mickey Mouse"
            },
            {
                "role": "system",
                "content": "Bugs Bunny,SpongeBob SquarePants,Tom and Jerry,Donald Duck,Pikachu,Scooby-Doo,Garfield,Dora the Explorer,Winnie the Pooh,Sonic the Hedgehog,The Simpsons,Elsa (Frozen),Shrek,Hello Kitty,Bart Simpson,Tweety Bird,Woody Woodpecker,Peter Griffin (Family Guy),Betty Boop,Charlie Brown"
            },
            {
                "role": "user",
                "content": "find 20  word or phrase of concepts about Bill Clinton"
            },
            {
                "role": "system",
                "content": "George Washington,Abraham Lincoln,Barack Obama,Thomas Jefferson,Ronald Reagan,John F. Kennedy,Theodore Roosevelt,Franklin D. Roosevelt,George W. Bush,Harry S. Truman,Dwight D. Eisenhower,Woodrow Wilson,Andrew Jackson,James Madison,James Monroe,Jimmy Carter,Gerald Ford,Lyndon B. Johnson,Richard Nixon,Herbert Hoover"
            },
            {
                "role": "user",
                "content": "find 20  word or phrase of concepts about Golf ball"
            },
            {
                "role": "system",
                "content": "Basketball,Tennis racket,Golf club,Baseball bat,Hockey stick,Football helmet,Cricket bat,Rugby ball,Bowling ball,Ping pong paddle,Lacrosse stick,Badminton shuttlecock,Volleyball net,Ski poles,Archery bow,Climbing harness,Boxing gloves,Swimming goggles,Dumbbells,Yoga mat"
            },
            {
                "role": "user",
                "content": f"find {num}  word or phrase of concepts about {ablated_concept}"
            },
        ]
    }

    response = requests.post(url, headers=headers, data=json.dumps(data).encode('utf-8'))
    result = response.content.decode("utf-8")
    print(result)
    return result


def get_synonym(ablated_concept, num):
    data = {
        "max_tokens": 1200,
        "model": "gpt-3.5-turbo",
        "temperature": 0.8,
        "top_p": 1,
        "presence_penalty": 1,
        "messages": [
            {
                "role": "system",
                "content": "You are ChatGPT, a large language model trained by OpenAI. Answer as concisely as possible."
            },
            {
                "role": "user",
                "content": "find 20  word or phrase of concepts about Mickey Mouse"
            },
            {
                "role": "system",
                "content": "Bugs Bunny,SpongeBob SquarePants,Tom and Jerry,Donald Duck,Pikachu,Scooby-Doo,Garfield,Dora the Explorer,Winnie the Pooh,Sonic the Hedgehog,The Simpsons,Elsa (Frozen),Shrek,Hello Kitty,Bart Simpson,Tweety Bird,Woody Woodpecker,Peter Griffin (Family Guy),Betty Boop,Charlie Brown"
            },
            {
                "role": "user",
                "content": "find 20  word or phrase of concepts about Bill Clinton"
            },
            {
                "role": "system",
                "content": "George Washington,Abraham Lincoln,Barack Obama,Thomas Jefferson,Ronald Reagan,John F. Kennedy,Theodore Roosevelt,Franklin D. Roosevelt,George W. Bush,Harry S. Truman,Dwight D. Eisenhower,Woodrow Wilson,Andrew Jackson,James Madison,James Monroe,Jimmy Carter,Gerald Ford,Lyndon B. Johnson,Richard Nixon,Herbert Hoover"
            },
            {
                "role": "user",
                "content": "find 20  word or phrase of concepts about Golf ball"
            },
            {
                "role": "system",
                "content": "George Washington,Abraham Lincoln,Barack Obama,Thomas Jefferson,Ronald Reagan,John F. Kennedy,Theodore Roosevelt,Franklin D. Roosevelt,George W. Bush,Harry S. Truman,Dwight D. Eisenhower,Woodrow Wilson,Andrew Jackson,James Madison,James Monroe,Jimmy Carter,Gerald Ford,Lyndon B. Johnson,Richard Nixon,Herbert Hoover"
            },
            {
                "role": "user",
                "content": f"find {num}  word or phrase of concepts about {ablated_concept}"
            },
        ]
    }

    response = requests.post(url, headers=headers, data=json.dumps(data).encode('utf-8'))
    result = response.content.decode("utf-8")
    print(result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='getConcept',
        description='get concept from gpt')
    parser.add_argument('--ablated_concept', help='prompt corresponding to concept to erase', type=str, required=True)
    # parser.add_argument('--category', type=str, required=True)
    parser.add_argument('--anchor_num', type=int, required=True)
    parser.add_argument('--synonym_num', type=int, required=True)
    args = parser.parse_args()
    anchor_txt = get_anchor_concepts(args.ablated_concept, args.anchor_num)
    print(anchor_txt)
    # synonym_txt = get_synonym(args.ablated_concept, args.synonym_num)
    # with open(f'data/{args.ablated_concept}_anchor_gpt.txt', 'w') as f:
    #     f.write(anchor_txt)
    # with open(f'data/{args.ablated_concept}_synonym_gpt.txt', 'w') as f:
    #     f.write(synonym_txt)
