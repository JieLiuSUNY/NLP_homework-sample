# author: Jie Liu (using part of code from data mining class which is provided by Professor Feng Chen before)

from sklearn import svm
from sklearn.model_selection import GridSearchCV
from numpy import array
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix


# load the data file into this code and separate the labels and sentence text
stopwords = ["a", "about", "above", "above", "across", "after", "afterwards", "again", "against", "all", "almost", "alone", "along", "already", "also","although","always","am","among", "amongst", "amoungst", "amount",  "an", "and", "another", "any","anyhow","anyone","anything","anyway", "anywhere", "are", "around", "as",  "at", "back","be","became", "because","become","becomes", "becoming", "been", "before", "beforehand", "behind", "being", "below", "beside", "besides", "between", "beyond", "bill", "both", "bottom","but", "by", "call", "can", "cannot", "cant", "co", "con", "could", "couldnt", "cry", "de", "describe", "detail", "do", "done", "down", "due", "during", "each", "eg", "eight", "either", "eleven","else", "elsewhere", "empty", "enough", "etc", "even", "ever", "every", "everyone", "everything", "everywhere", "except", "few", "fifteen", "fify", "fill", "find", "fire", "first", "five", "for", "former", "formerly", "forty", "found", "four", "from", "front", "full", "further", "get", "give", "go", "had", "has", "hasnt", "have", "he", "hence", "her", "here", "hereafter", "hereby", "herein", "hereupon", "hers", "herself", "him", "himself", "his", "how", "however", "hundred", "ie", "if", "in", "inc", "indeed", "interest", "into", "is", "it", "its", "itself", "keep", "last", "latter", "latterly", "least", "less", "ltd", "made", "many", "may", "me", "meanwhile", "might", "mill", "mine", "more", "moreover", "most", "mostly", "move", "much", "must", "my", "myself", "name", "namely", "neither", "never", "nevertheless", "next", "nine", "no", "nobody", "none", "noone", "nor", "not", "nothing", "now", "nowhere", "of", "off", "often", "on", "once", "one", "only", "onto", "or", "other", "others", "otherwise", "our", "ours", "ourselves", "out", "over", "own","part", "per", "perhaps", "please", "put", "rather", "re", "same", "see", "seem", "seemed", "seeming", "seems", "serious", "several", "she", "should", "show", "side", "since", "sincere", "six", "sixty", "so", "some", "somehow", "someone", "something", "sometime", "sometimes", "somewhere", "still", "such", "system", "take", "ten", "than", "that", "the", "their", "them", "themselves", "then", "thence", "there", "thereafter", "thereby", "therefore", "therein", "thereupon", "these", "they", "thickv", "thin", "third", "this", "those", "though", "three", "through", "throughout", "thru", "thus", "to", "together", "too", "top", "toward", "towards", "twelve", "twenty", "two", "un", "under", "until", "up", "upon", "us", "very", "via", "was", "we", "well", "were", "what", "whatever", "when", "whence", "whenever", "where", "whereafter", "whereas", "whereby", "wherein", "whereupon", "wherever", "whether", "which", "while", "whither", "who", "whoever", "whole", "whom", "whose", "why", "will", "with", "within", "without", "would", "yet", "you", "your", "yours", "yourself", "yourselves", "the", "I"]
sentences = []
for line in open('amazon_labelled.txt').readlines():
    items = line.split('\t')
    items[1]= items[1].strip('\n')
    sentences.append([int(items[1]), items[0].lower().strip()])



# compute the frequency of each word and find out the most frequent words(frequency >=8)
vocab = dict()
for class_label, text in sentences:
    for term in text.split():
        term = term.lower()
        if len(term) > 2 and term not in stopwords:
            if vocab.has_key(term):
                vocab[term] = vocab[term] + 1
            else:
                vocab[term] = 1

vocab = {term: freq for term, freq in vocab.items() if freq > 8}
vocab = {term: idx for idx, (term, freq) in enumerate(vocab.items())}
print len(vocab)


# Generate X and y
X = []
y = []
for class_label, text in sentences:
    x = [0] * len(vocab)
    terms = [term for term in text.split() if len(term) > 2]
    for term in terms:
        if vocab.has_key(term):
            x[vocab[term]] += 1
    y.append(class_label)
    X.append(x)


# using SVM
clf1 = svm.SVC(kernel='linear', C=1.0)
clf1.fit(X, y)
print "Train: SVM Model accuracy = ", clf1.score(X,y)



# using 10 folder cross validation
svc = svm.SVC(kernel='linear')
Cs = range(1, 20)
clf2 = GridSearchCV(estimator=svc, param_grid=dict(C=Cs), cv = 10)
clf2.fit(X, y)
print "Train: ten foler cross validation Model accuracy = ", clf2.score(X,y)



# load test sentences and generate the test X; saving the real label of each sentence
sentense_test = []
y_real = []
for line in open('amazon_test.txt').readlines():
    items = line.split('\t')
    sentense_test.append(items[0])
    y_real.append(int(items[1].strip()))

X_test = []
for text in sentense_test:
    x = [0] * len(vocab)
    terms = [term for term in text.split() if len(term) > 2]
    for term in terms:
        if vocab.has_key(term):
            x[vocab[term]] += 1
    X_test.append(x)

# using SVM and ten folder cross validation to predit the rest sentences and compute their accuracy respectively.
y_real = array(y_real)
n1=n2=0
y1 = clf1.predict(X_test)
y2= clf2.predict(X_test)
#print y1
#print y_real

print "For SVM: "
print "F1 score: ",(f1_score(y_real, y1, average="macro"))
print "Precision score: ",(precision_score(y_real, y1, average="macro"))
print "Recall score: ",(recall_score(y_real, y1, average="macro")), '\n'

print "For ten folder cross validation: "
print "F1 score: ",(f1_score(y_real, y2, average="macro"))
print "Precision score: ",(precision_score(y_real, y2, average="macro"))
print "Recall score: ",(recall_score(y_real, y2, average="macro"))




