import csv
import sys

from util import Node, StackFrontier, QueueFrontier

# Maps names to a set of corresponding person_ids
names = {}

# Maps person_ids to a dictionary of: name, birth, movies (a set of movie_ids)
people = {}

# Maps movie_ids to a dictionary of: title, year, stars (a set of person_ids)
movies = {}


def load_data(directory):
    """
    Load data from CSV files into memory.
    """
    # Load people
    with open(f"{directory}/people.csv", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            people[row["id"]] = {
                "name": row["name"],
                "birth": row["birth"],
                "movies": set()
            }
            if row["name"].lower() not in names:
                names[row["name"].lower()] = {row["id"]}
            else:
                names[row["name"].lower()].add(row["id"])

    # Load movies
    with open(f"{directory}/movies.csv", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            movies[row["id"]] = {
                "title": row["title"],
                "year": row["year"],
                "stars": set()
            }

    # Load stars
    with open(f"{directory}/stars.csv", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                people[row["person_id"]]["movies"].add(row["movie_id"])
                movies[row["movie_id"]]["stars"].add(row["person_id"])
            except KeyError:
                pass


def main():
    if len(sys.argv) > 2:
        sys.exit("Usage: python degrees.py [directory]")
    directory = sys.argv[1] if len(sys.argv) == 2 else "large"

    # Load data from files into memory
    print("Loading data...")
    load_data(directory)
    print("Data loaded.")

    source = person_id_for_name(input("Name: "))
    if source is None:
        sys.exit("Person not found.")
    target = person_id_for_name(input("Name: "))
    if target is None:
        sys.exit("Person not found.")

    path = shortest_path(source, target)

    if path is None:
        print("Not connected.")
    else:
        degrees = len(path)
        print(f"{degrees} degrees of separation.")
        path = [(None, source)] + path
        for i in range(degrees):
            person1 = people[path[i][1]]["name"]
            person2 = people[path[i + 1][1]]["name"]
            movie = movies[path[i + 1][0]]["title"]
            print(f"{i + 1}: {person1} and {person2} starred in {movie}")


def shortest_path(source, target):
    """
    Returns the shortest list of (movie_id, person_id) pairs
    that connect the source to the target.

    If no possible path, returns None.
    """

    path = [] # Holds the path followed to reach the target

    # if the source and target are the same
    if source == target:
        print("Source and target are the same")
        return None


    # Initialize frontier to be have the start as the source id
    start = Node(state=source, movie_id="Rodney", neighbors=neighbors_for_person(source), parent=None)

    #Frontier where to start the searching implemented as a queue
    frontier = QueueFrontier()
    frontier.add(start)

    # a set of already explored persons
    explored = set()

    # keep looping until solution is found
    while True:
        # returns None if there is no solution
        if frontier.empty():
            return None
        
        # choose a node from the frontier
        node = frontier.remove()

        # #check if the node is the solution
        # if node.state == target:
        #     pass

        # marking person as explored
        explored.add(node.state)

        #node has no neighbors so we skip to next node
        if len(node.neighbors) == 0:
            continue
        
        # checks whether neighbors is the target
        for character in node.neighbors:
            if character[1] == target:
                # target is found
                path.append(character)
                while node.parent is not None:
                    # print(node.state)
                    path.append((node.movie_id, node.state))
                    node = node.parent
                print("This is the path", path)
                # path.reverse() #reverses the list to start from scratch
                return path

        #if not target we add them to the frontier
        for character in node.neighbors:
            # if character is already explored and is in frontier skip adding from the frontier
            if (character[1] in explored) or (frontier.contains_state(character[1])):
                continue

            new_node = Node(state=character[1], movie_id=character[0], neighbors=neighbors_for_person(character[1]),
                        parent=node)
            frontier.add(new_node)

def person_id_for_name(name):
    """
    Returns the IMDB id for a person's name,
    resolving ambiguities as needed.
    """
    person_ids = list(names.get(name.lower(), set()))
    if len(person_ids) == 0:
        return None
    elif len(person_ids) > 1:
        print(f"Which '{name}'?")
        for person_id in person_ids:
            person = people[person_id]
            name = person["name"]
            birth = person["birth"]
            print(f"ID: {person_id}, Name: {name}, Birth: {birth}")
        try:
            person_id = input("Intended Person ID: ")
            if person_id in person_ids:
                return person_id
        except ValueError:
            pass
        return None
    else:
        return person_ids[0]


def neighbors_for_person(person_id):
    """
    Returns (movie_id, person_id) pairs for people
    who starred with a given person.
    """
    movie_ids = people[person_id]["movies"]
    neighbors = set()
    for movie_id in movie_ids:
        for person_id in movies[movie_id]["stars"]:
            neighbors.add((movie_id, person_id))
    # print("Calling from neighbors", neighbors)
    return neighbors


if __name__ == "__main__":
    main()
