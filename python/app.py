from csv import DictWriter
from datetime import datetime
from termcolor import colored
import pandas as pd
import dill as pickle
pd.options.mode.chained_assignment = None

# save dataset
recipe = pd.read_csv("../dataset/recipes.csv")
recipe = recipe.astype({"RecipeId": str})
review = pd.read_csv("../dataset_2/predicted_all_comments.csv")
users = pd.read_csv("../dataset_2/UserId_Password.csv")

#  save model to text mining
count_vect_model = pickle.load(open('../model/count_vect.sav', 'rb'))
tfidf_model = pickle.load(open('../model/tfidf_model.sav', 'rb'))
loaded_model = pickle.load(open("../model/static_model.sav", 'rb'))

aut_id = 0
aut_name = ""


def insertNewComment(id):
    global review
    global aut_id
    global aut_name

    print(colored("***********************************", "blue"))
    new_comm = input("Insert new comment: ")
    print(colored("***********************************", "blue"))

    #  Set attribute value
    last_id = review['ReviewId'].max()
    new_id = last_id + 1

    date = datetime.today().strftime('%Y-%m-%d')

    new_comm = [new_comm]

    #  Assign the predicted sentiment
    x_test_counts = count_vect_model.transform(new_comm)
    x_test_tfidf = tfidf_model.transform(x_test_counts)
    predicted = loaded_model.predict(x_test_tfidf)

    print("The predicted sentiment of your comment is: " + predicted[0])

    new_row = {'ReviewId': new_id, 'RecipeId': id, 'AuthorId': aut_id, 'AuthorName': aut_name, 'Rating': predicted[0],
               'Review': new_comm[0], 'DateSubmitted': date, 'DateModified': date}

    #  Modify dataframe to memorize the new comment
    review = review.append(new_row, ignore_index=True)

    print("Comment entered successfully.")

    #  Modify the csv File for a permanent modification
    headersCSV = ['ReviewId', 'RecipeId', 'AuthorId', 'AuthorName', 'Rating', 'Review', 'DateSubmitted', 'DateModified']
    with open('../dataset_2/predicted_all_comments.csv', 'a', newline='') as f_object:
        dictwriter_object = DictWriter(f_object, fieldnames=headersCSV)
        dictwriter_object.writerow(new_row)
        f_object.close()


def viewRecipe(row, id):
    global review
    global aut_id
    global aut_name

    #  MENU
    while True:
        str_id=str(id)
        print()
        print(colored("***********************************", "green"))
        print(colored("RECIPE MENU' ID:"+ str_id, "green"))
        print("What do you want to do?")
        print("Select:")
        print("1 -> View Information")
        print("2 -> View comments")
        print("3 -> View general sentiment about it")
        print("4 -> Insert new comment")
        print("0 -> Previous menu")
        print(colored("***********************************", "green"))

        print(colored("***********************************", "blue"))
        command = input("Write command: ")
        try:
            command = int(command)
        except ValueError:
            print("ATTENTION! Insert a number.")
            continue
        print(colored("***********************************", "blue"))

        if command == 0:
            break

        if command != 1 and command != 2 and command != 3 and command != 4:
            print("ATTENTION! Wrong command")
            continue

        #  View information
        if command == 1:
            print("INFORMATION RECIPE")
            print("Title: " + row['Name'].values)
            print("Author Name: " + row['AuthorName'].values)
            print("Category: " + row['RecipeCategory'].values)
            print("Ingredients: " + row['RecipeIngredientParts'].values)
            print("Instruction: " + row['RecipeInstructions'].values)

        #  View Comments
        if command == 2:
            #  Extract recipe's comments
            id = int(id)
            comments = review.loc[(review['RecipeId'] == id)]
            if len(comments) == 0:
                print("Comments do not present for this recipe.")
                continue

            count = len(comments)  # How many comments are present
            count = str(count)
            print(count + " comments are present.")

            comments.sort_values(by=['DateModified'], ascending=False, inplace=True)  # sort the dataframe by date
            print("Author Name -> Associated sentiment -> Text")
            print(comments['AuthorName'][:10] + " -> " + comments['Rating'][:10] + " -> " + comments['Review'][:10])
            i = 10

            while i < len(comments):
                print()
                print(colored("***********************************", "green"))
                print(colored("COMMENT MENU':", "green"))
                print("What do you want to do?")
                print("Select:")
                print("1 -> Next 10 comments...")
                print("0 -> Previous menu")
                print(colored("***********************************", "green"))
                print()

                print(colored("***********************************", "blue"))
                command = input("Write command: ")
                try:
                    command = int(command)
                except ValueError:
                    print("ATTENTION! Insert a number.")
                    continue
                print(colored("***********************************", "blue"))

                if command == 0:
                    break

                if command != 1:
                    print("ATTENTION! Wrong command")
                    continue

                if command == 1:
                    print("Author Name -> Associated sentiment -> Text")
                    print(comments['AuthorName'][i:i + 10] + " -> " + comments['Rating'][i:i + 10] + " -> " + comments['Review'][i:i + 10])
                    i = i + 10

            print()
            print("Finished comments")

        #  View sentiments
        if command == 3:
            #  Extract recipe's comments
            id = int(id)
            comments = review.loc[(review['RecipeId'] == id)]
            if len(comments) == 0:
                print("Comments do not present for this recipe.")
                continue

            count = len(comments)  # How many comments are present
            count = str(count)
            print(count + " comments are present.")

            all_sentiment = comments.Rating.value_counts()
            try:
                neg = str(all_sentiment['negative'])
                print(neg + " comments are NEGATIVE.")
            except KeyError:
                print("0 comments are NEGATIVE.")

            try:
                neu = str(all_sentiment['neutral'])
                print(neu + " comments are NEUTRAL.")
            except KeyError:
                print("0 comments are NEUTRAL.")

            try:
                p = str(all_sentiment['positive'])
                print(p + " comments are POSITIVE.")
            except KeyError:
                print("0 comments are POSITIVE")

        #  Insert new comment
        if command == 4:
            insertNewComment(id)


def browseRecipe(df):
    if len(df) == 0:
        global recipe
        use_df = recipe
    else:
        use_df = df


    i = 10  # index to browse 10 recipe at time

    #  Print the first 10 recipes
    print(" Recipe ID -> Recipe name -> Author name")
    print(use_df['RecipeId'][:10] + " -> " + use_df['Name'][:10] + " -> " + use_df['AuthorName'][:10])

    while True:

        print()
        print(colored("***********************************", "green"))
        print(colored("RECIPE MENU'", "green"))
        print("What do you want to do?")
        print("Select:")
        print("1 -> Next 10 recipes...")
        print("2 -> View Recipe")
        print("0 -> Exit")
        print(colored("***********************************", "green"))
        print()

        print(colored("***********************************", "blue"))
        command = input("Write command: ")
        try:
            command = int(command)
        except ValueError:
            print("ATTENTION! Insert a number.")
            continue
        print(colored("***********************************", "blue"))

        if command == 0:
            break

        if command != 1 and command != 2:
            print("ATTENTION! Wrong command")
            continue

        #  Scroll recipes
        if command == 1:
            if i < len(use_df):
                print("Recipe name -> Author name")
                print(use_df['RecipeId'][i:i+10] + " -> " + use_df['Name'][i:i+10] + " -> " + use_df['AuthorName'][i:i+10])
            else:
                print("There are no more recipes.")
                continue
            i = i + 10

        #  Visualize recipe
        if command == 2:
            while True:
                print()
                print(colored("***********************************", "blue"))
                id_recipe = input("Write desired recipe ID: ")
                print(colored("***********************************", "blue"))
                print()

                row = use_df.loc[(use_df['RecipeId'] == id_recipe)]
                if len(row) == 0:
                    print("ATTENTION! ID does not exist!")
                    continue
                else:
                    break

            viewRecipe(row, id_recipe)


def browseUser():
    global recipe

    i = 10  # index to browse 10 recipe at time

    recipe_str = recipe.astype({"AuthorId": str})
    users = recipe_str.groupby(['AuthorId', 'AuthorName'], as_index=False).count()
    users_str = users.astype({"RecipeId": str})

    print("ID Author -> Author Name -> Number of recipes")
    print(users_str['AuthorId'][:10] + " -> " + users_str['AuthorName'][:10] + " -> " + users_str['RecipeId'][:10])

    while i < len(users_str):

        print()
        print(colored("***********************************", "green"))
        print(colored("USER MENU'", "green"))
        print("What do you want to do?")
        print("Select:")
        print("1 -> Next 10 users...")
        print("2 -> View user's recipes")
        print("0 -> Exit")
        print(colored("***********************************", "green"))
        print()

        print(colored("***********************************", "blue"))
        command = input("Write command: ")
        try:
            command = int(command)
        except ValueError:
            print("ATTENTION! Insert a number.")
            continue
        print(colored("***********************************", "blue"))

        if command == 0:
            print("*** PRINCIPAL MENU ***")
            break

        if command != 1 and command != 2:
            print("ATTENTION! Wrong command")
            continue

        #  Scroll users
        if command == 1:
            print("ID Author -> Author Name -> Number of recipes")
            print(users_str['AuthorId'][i:i+10] + " -> " + users_str['AuthorName'][i:i+10] + " -> " + users_str['RecipeId'][i:i+10])
            i = i + 10

        #  view user's recipes
        if command == 2:
            while True:
                print()
                print(colored("***********************************", "blue"))
                id_users = str(input("Write desired user ID: "))
                print(colored("***********************************", "blue"))
                print()

                user_recipe = recipe.loc[(recipe_str['AuthorId'] == id_users)]
                if len(user_recipe) == 0:
                    print("ATTENTION! ID does not exist!")
                    continue
                else:
                    browseRecipe(user_recipe)
                    break


def browseCategory():
    global recipe

    categories = recipe.groupby(['RecipeCategory']).size().reset_index(name='counts')
    categories = categories.sort_values(by=['counts'], ascending=False)

    categories_str = categories.astype({"counts": str})

    #  Print the first 10 recipes
    print("Recipe's categories -> Number of recipes")
    print(categories_str['RecipeCategory'][:10] + " -> " + categories_str['counts'][:10])
    i = 10

    while i < len(categories_str):

        print()
        print(colored("***********************************", "green"))
        print(colored("CATEGORY MENU'", "green"))
        print("What do you want to do?")
        print("Select:")
        print("1 -> Next 10 categories...")
        print("2 -> View recipes with a specified category")
        print("0 -> Exit")
        print(colored("***********************************", "green"))
        print()

        print(colored("***********************************", "blue"))
        command = input("Write command: ")
        try:
            command = int(command)
        except ValueError:
            print("ATTENTION! Insert a number.")
            continue
        print(colored("***********************************", "blue"))

        if command == 0:
            break

        if command != 1 and command != 2:
            print("ATTENTION! Wrong command")
            continue

        #  Scroll users
        if command == 1:
            print("Recipe's categories -> Number of recipes")
            print(categories_str['RecipeCategory'][i:i+10] + " -> " + categories_str['counts'][i:i+10])
            i = i + 10

        #  view recipes with a specified category
        if command == 2:
            while True:
                print()
                print(colored("***********************************", "blue"))
                cat = str(input("Write desired category: "))
                print(colored("***********************************", "blue"))
                print()

                cat_recipe = recipe.loc[(recipe['RecipeCategory'] == cat)]
                if len(cat_recipe) == 0:
                    print("ATTENTION! Category does not exist!")
                    continue
                else:
                    browseRecipe(cat_recipe)
                    break


def control_login():
    global users
    global aut_name
    global aut_id

    while True:
        print(colored("***********************************", "green"))
        print(colored("LOGIN MENU'", "green"))
        print("What do you want to do?")
        print("Select:")
        print("1-> Log In")
        print("2-> Register")
        print(colored("***********************************", "green"))

        print(colored("***********************************", "blue"))
        command = input("Write command: ")
        try:
            command = int(command)
        except ValueError:
            print("ATTENTION! Insert a number.")
            continue
        print(colored("***********************************", "blue"))
        print()

        if command != 1 and command != 2 and command != 3:
            print("ATTENTION! Wrong command")
            continue

        #  Log in
        if command == 1:
            print()
            print(colored("***********************************", "blue"))
            while True:
                userId = input("User ID: ")
                try:
                    userId = int(userId)
                except ValueError:
                    print("Insert a number as UserId")
                    continue
                break
            while True:
                password = input("password: ")
                try:
                    password = int(password)
                    break
                except ValueError:
                    print("Insert a number as Password")
                    continue
            print(colored("***********************************", "blue"))
            print()
            out = users.loc[((users['UserId'] == userId) & (users['Password'] == password))]

            if out.empty:
                print("ATTENTION! Wrong username or password.")
                continue
            else:
                aut_id = userId
                aut_name = out['UserName'].item()
                return

        if command == 2:
            print()
            print(colored("***********************************", "blue"))
            print("Choose an user ID and password:")
            while True:
                new_userId = input("User ID: ")
                try:
                    new_userId = int(new_userId)
                except ValueError:
                    print("Insert a number as UserId")
                    continue
                break
            new_name = input("User name: ")
            while True:
                new_password = input("password: ")
                try:
                    new_password = int(new_password)
                    break
                except ValueError:
                    print("Insert a number as Password")
                    continue
            print(colored("***********************************", "blue"))
            print()

            out = users.loc[users['UserId'] == new_userId]

            if out.empty:
                new_row = {'UserId': new_userId, 'UserName': new_name, 'Password': new_password}
                # Modify dataframe to memorize the new User
                users = users.append(new_row, ignore_index=True)

                #  Modify the csv File for a permanent modification
                headersCSV = ['UserId', 'UserName', 'Password']
                with open('../dataset_2/UserId_Password.csv', 'a', newline='') as f_object:
                    dictwriter_object = DictWriter(f_object, fieldnames=headersCSV)
                    dictwriter_object.writerow(new_row)
                    f_object.close()

                print("You have successfully registered!")
                aut_id = new_userId
                aut_name = new_name
                return
            else:
                print("Your User ID Already Exists.")
                continue


if __name__ == "__main__":

    print(colored("WELCOME TO RECYLIS", "magenta"))
    control_login()

    command = 1
    while True:
        print()
        print(colored("***********************************", "green"))
        print(colored("PRINCIPAL MENU'", "green"))
        print("What do you want to do?")
        print("Select:")
        print("1-> Browse all the recipes")
        print("2-> Browse all the users")
        print("3-> Browse categories")
        print("0-> exit")
        print(colored("***********************************", "green"))

        print(colored("***********************************", "blue"))
        command = input("Write command: ")
        try:
            command = int(command)
        except ValueError:
            print("ATTENTION! Insert a number.")
            continue
        print(colored("***********************************", "blue"))
        print()

        if command == 0:
            print(colored("GOODBY", "magenta"))
            break

        if command != 1 and command != 2 and command != 3:
            print("ATTENTION! Wrong command")
            continue

        if command == 1:
            browseRecipe([])
        elif command == 2:
            browseUser()
        else:
            browseCategory()



