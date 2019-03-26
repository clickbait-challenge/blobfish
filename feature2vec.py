class feature2vec:
    """ This class allows to trasform a sentence in a vector. Every feature is  transformed in a number
    """

    vect = []

    # Vector of the class

    def __init__(self, tokens_pos):
        """
        Parameters
        ----------
        token_pos : list of list of tuples
            this list is a list of all tuples, as result of tokenization and pos tagging.
            e.g. a list of [('NATO', 'NNP'), ('and', 'CC'), ('Russia', 'NNP'), ('are', 'VBP'), ('in', 'IN'), ('a', 'DT'), ('game', 'NN'), ('of', 'IN'), ('cat', 'NN'), ('and', 'CC'), ('mouse', 'NN'), ('in', 'IN'), ('the', 'DT'), ('Baltic', 'NNP'), ('skies', 'NNS')]

        """
        self.tokens_pos = tokens_pos

    def count_POS(self, array, type):
        """ Count number of a given type of POS
        Parameters
        ----------
        array : list
            the list of tuples of the sentence
        type : string
            Part of Speech tag used by Penn Treebank project
        """
        count = 0
        if not array:
            return 0
        else:
            for a in array:
                if a[1] == type:
                    count = count + 1
            return count

    def count_tokens(self, array):
        """ Return the length of the array
        Parameters
        ----------
        array : list
            the list of tuples of the sentence

        """
        return len(array)

    def AWL(self, array):
        """ Calculate the Average Word Length
        Parameters
        ----------
        array : list
            the list of tuples of the sentence

        """
        avg = 0
        for a in array:
            avg = avg + len(a[0])
        if len(array) != 0:
            avg = avg / len(array)
        else:
            avg = 0
        return avg

    def longest_word(self, array):
        """ Return the longest word in the sentence
        Parameters
        ----------
        array : list
            the list of tuples of the sentence

        """
        longest = 0
        for a in array:
            if len(a[0]) >= longest:
                longest = len(a[0])
        return longest

    def count_big_POS(self, array, type1, type2):
        """ Count the number of occurrence of bigram of POS
        Parameters
        ----------
        array : list
            the list of tuples of the sentence
        type1 : string
            Part of Speech tag used by Penn Treebank project
        type2 : string
            Part of Speech tag used by Penn Treebank project
        """
        bigCount = 0
        count = 0
        if type1 == type2:
            for a in array:
                if a[1] == type1:
                    if count == 1:
                        bigCount = bigCount + 1
                    else:
                        count = count + 1
                else:
                    count = 0
            return bigCount
        else:
            for a in array:
                if a[1] == type1:
                    count = 1
                if count == 1 and a[1] != type2 and a[1] != type1:
                    count = 0
                if a[1] == type2 and count == 1:
                    bigCount = bigCount + 1
            return bigCount

    def count_tri_POS(self, array, type1, type2, type3):
        """ Count the number of occurrence of trigram of POS
        Parameters
        ----------
        array : list
            the list of tuples of the sentence
        type1 : string
            Part of Speech tag used by Penn Treebank project
        type2 : string
            Part of Speech tag used by Penn Treebank project
        type3 : string
            Part of Speech tag used by Penn Treebank project
        """
        count = 0
        triCount = 0
        if type1 == type2 and type1 == type3:
            for a in array:
                if a[1] == type1:
                    if count == 2:
                        triCount = triCount + 1
                    else:
                        count = count + 1
                else:
                    count = 0
            return triCount
        elif type1 == type2:
            for a in array:
                if a[1] == type1:
                    if count == 1:
                        count = count + 1
                    else:
                        count = count + 1
                elif a[1] == type3:
                    if count == 2:
                        triCount = triCount + 1
                    else:
                        count = 0
                else:
                    count = 0
            return triCount
        elif type1 == type3:
            for a in array:
                if a[1] == type1:
                    if count == 2:
                        triCount = triCount + 1
                        count = 1
                    else:
                        count = 1
                if a[1] == type2:
                    if count == 1:
                        count = count + 1
                    else:
                        count = 0
            return triCount
        elif type2 == type3:
            for a in array:
                if a[1] == type1:
                    count = 1
                elif a[1] == type2:
                    if count == 1:
                        count = count + 1
                    if count == 2:
                        triCount = triCount + 1
                        count = 0
                else:
                    count = 0
            return triCount
        else:
            for a in array:
                if a[1] == type1:
                    count = 1
                elif a[1] == type2:
                    if count == 1:
                        count = count + 1
                    else:
                        count = 0
                elif a[1] == type3:
                    if count == 2:
                        triCount = triCount + 1
                        count = 0
                    else:
                        count = 0
                else:
                    count = 0
            return triCount

    def start_with(self, array, type):
        """ Return 1 if the sentence start witj a specific POS tag
        Parameters
        ----------
        array : list
            the list of tuples of the sentence
        type : string
            Part of Speech tag used by Penn Treebank project
        """
        if not array:
            return 0
        else:
            if array[0][1] == type:
                return 1
            else:
                return 0

    def feature2vec(self, array):
        """
        Feature vector
        [N 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0] Number of NNP (int)
        [0 N 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0] Number of tokens (int)
        [0 0 N 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0] Number of POS 2-gram NNP NNP (int)
        [0 0 0 N 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0] Whether the post start with number (1 true, 0 false)
        [0 0 0 0 N 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0] Number of IN (int)
        [0 0 0 0 0 N 0 0 0 0 0 0 0 0 0 0 0 0 0 0] Number of POS 2-gram NNP VBZ (int)
        [0 0 0 0 0 0 N 0 0 0 0 0 0 0 0 0 0 0 0 0] Number of POS 2-gram IN NNP (int)
        [0 0 0 0 0 0 0 N 0 0 0 0 0 0 0 0 0 0 0 0] Number of WRB (int)
        [0 0 0 0 0 0 0 0 N 0 0 0 0 0 0 0 0 0 0 0] Number of NN (int)
        [0 0 0 0 0 0 0 0 0 N 0 0 0 0 0 0 0 0 0 0] Average word length (float)
        [0 0 0 0 0 0 0 0 0 0 N 0 0 0 0 0 0 0 0 0] Length of the longest word (int)
        [0 0 0 0 0 0 0 0 0 0 0 N 0 0 0 0 0 0 0 0] Number of PRP (int)
        [0 0 0 0 0 0 0 0 0 0 0 0 N 0 0 0 0 0 0 0] Number of VBZ (int)
        [0 0 0 0 0 0 0 0 0 0 0 0 0 N 0 0 0 0 0 0] Number of POS 3-gram NNP NNP VBZ (int)
        [0 0 0 0 0 0 0 0 0 0 0 0 0 0 N 0 0 0 0 0] Number of POS 2-gram NN IN (int)
        [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 N 0 0 0 0] Number of POS 3-gram NN IN NNP (int)
        [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 N 0 0 0] Number of POS 2-gram NNP . (int)
        [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 N 0 0] Number of POS 2-gram PRP VBP (int)
        [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 N 0] Number of WP (int)
        [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 N] Number of DT (int)
        """

        vec = []
        vec.append(self.count_POS(array, "NNP"))
        vec.append(self.count_tokens(array))
        vec.append(self.count_big_POS(array, "NNP", "NNP"))
        vec.append(self.start_with(array, "CD"))
        vec.append(self.count_POS(array, "IN"))
        vec.append(self.count_big_POS(array, "NNP", "VBZ"))
        vec.append(self.count_big_POS(array, "IN", "NNP"))
        vec.append(self.count_POS(array, "WRB"))
        vec.append(self.count_POS(array, "NN"))
        vec.append(self.AWL(array))
        vec.append(self.longest_word(array))
        vec.append(self.count_POS(array, "PRP"))
        vec.append(self.count_POS(array, "VBZ"))
        # vec.append(self.count_tri_POS(array, "NNP", "NNP", "VBZ"))  # to leave
        vec.append(self.count_big_POS(array, "NN", "IN"))
        vec.append(self.count_tri_POS(array, "NN", "IN", "NNP"))
        # vec.append(self.count_big_POS(array, "NNP", "."))  # to leave
        # vec.append(self.count_big_POS(array, "PRP", "VBP"))  # to leave
        # vec.append(self.count_POS(array, "WP"))  # to leave
        vec.append(self.count_POS(array, "DT"))

        # vec.append(self.count_big_POS(array, "NNP", "IN"))  # to leave
        vec.append(self.count_tri_POS(array, "IN", "NNP", "NNP"))
        vec.append(self.count_POS(array, "POS"))
        vec.append(self.count_big_POS(array, "IN", "NN"))
        vec.append(self.count_POS(array, ","))
        # vec.append(self.count_big_POS(array, "NNP", "NNS"))  # to leave
        # vec.append(self.count_big_POS(array, "IN", "JJ"))  # to leave

        # vec.append(self.count_big_POS(array, "NNP", "POS"))  # to leave
        # vec.append(self.count_POS(array, "WDT"))  # to leave
        vec.append(self.count_big_POS(array, "NN", "NN"))
        # vec.append(self.count_big_POS(array, "NN", "NNP"))  # to leave
        # vec.append(self.count_big_POS(array, "NNP", "VBD"))  # to leave
        vec.append(self.count_POS(array, "RB"))

        # vec.append(self.count_tri_POS(array, "NNP", "NNP", "NNP"))  # to leave
        # vec.append(self.count_tri_POS(array, "NNP", "NNP", "NN"))  # to leave

        # vec.append(self.count_POS(array, "RBS"))  # to leave
        vec.append(self.count_POS(array, "VBN"))
        # vec.append(self.count_big_POS(array, "VBN", "IN"))  # to leave
        # vec.append(self.count_big_POS(array, "JJ", "NNP"))  # to leave
        vec.append(self.count_tri_POS(array, "NNP", "NN", "NN"))
        vec.append(self.count_big_POS(array, "DT", "NN"))

        self.vect.append(vec)

    def execute(self):
        for pos in self.tokens_pos:
            self.feature2vec(pos)
        print("in execute - vect len: ", len(self.vect))

    def get_vect(self):
        return self.vect