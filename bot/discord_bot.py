import asyncio
from random import randint

import discord

from evaluater import generate

client = discord.Client()
# path of model
generator = generate.TextGenerator(model_path=r"..\models\trainer_1\krishna\trash\final_model.hdf5")
with open("token.txt", "r") as f:  # REMEMBER: DON'T UPLOAD TOKEN.TXT TO GITHUB
    token = f.read()
    f.close()


def listify(text):
    list = []
    for char in text:
        list.append(char)
    return list


def purify_message(text=" ", users=[], charPairs=[], emptyReplace=None):
    # text: message text, users: list of users to remove, charPairs: list of characters to replace with pairs
    # emptyReplace: string to replace empty string
    for user in users:
        text = text.replace("@" + user.name, "")
    for charPair in charPairs:
        while charPair[0] in text:
            text = text.replace(charPair[0], charPair[1])
    if text == "":
        text = emptyReplace
    return text


@client.event
async def on_ready():
    print("ready")


@client.event
async def on_message(message):
    if message.author == client.user:
        pass
    else:
        if client.user in message.mentions:
            text = message.clean_content
            text = purify_message(text=text, users=[client.user], charPairs=[["@", ""], ["  ", " "]])
            seed_chars = listify(text)
            async with message.channel.typing():
                num_messages = randint(1, 10)
                responses, trash = generator.generate_messages(num_messages=num_messages, seed_chars=seed_chars,
                                                               min_seed_chars_generated=0,
                                                               max_seed_chars_before_delimiter=0, print_progress=False)
                print(text, num_messages)
                for response in responses:
                    response = purify_message(text=response, emptyReplace=". . .")
                    # limit response speed
                    await asyncio.sleep(1)
                    await message.channel.send(response)


# token
client.run(token)
