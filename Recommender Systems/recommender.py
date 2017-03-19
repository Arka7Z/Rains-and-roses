import csv
from math import sqrt
ratings={}
tRatings={}

def loadMovieLens (path='/home/amanpurwar/Desktop/MovieRecommender'):
	file=csv.reader(open('movies.csv','r'))
	movies={}
	for row in file:
		id,title=row[0:2]        #takes only the title and movie 
		movies[id]=title
	#ratings={}
	file=csv.reader(open('ratings.csv','r'))
	i=0
	for row in file:
		userId,movieId,rating=row[0:3]
		if rating=='rating':continue
		ratings.setdefault(userId,{})
		ratings[userId][movies[movieId]]=float(rating)
		
	return ratings

def transformRating(ratings):       #for item based collaborative filterings
	#tRatings={}
	for user in ratings:
		for movie in ratings[user]:
			tRatings.setdefault(movie,{})
			tRatings[movie][user]=ratings[user][movie]
	return tRatings

def pearson_similarity(ratings,movie1,movie2):
	common={}

	for user in tRatings[movie1]:
		if user in tRatings[movie2]:
			common[user]=1
	if len(common)==0:return 0
	

	sum1=sum([ratings[movie1][user] for user in common])
	sum2=sum([ratings[movie2][user] for user in common])
	sum12=(sum([ratings[movie1][u]*ratings[movie2][u] for u in common]))
	sum1Sq=(sum([pow(ratings[movie1][u],2) for u in common]))
	sum2Sq=sum([pow(ratings[movie2][u],2) for u in common])
	n=len(common)
	num=sum12-((sum1*sum2)/n)
	den=sqrt(sum1Sq-pow(sum1,2)/n)*sqrt(sum2Sq-pow(sum2,2)/n)
	if den==0:return 0
	r=num/den
	return r

def topMatches(ratings,movie,n=10):                          #returns n top matches for a particular movie
	#tRatings=transformRatings(ratings)
	matches=([(pearson_similarity(ratings,movie,other),other) for other in tRatings if other!=movie])
	matches.sort()
	matches.reverse()
	return matches[0:n]

def build_similaritems(ratings,n=10):
	similar={}                                 #similar[movie]=n movies movie is found similar to
	tRatings=transformRating(ratings)
	for movie in tRatings:
		similar_movie_list=topMatches(tRatings,movie,n)
		similar[movie]=similar_movie_list
	return similar


def getMovieRecommendations(ratings,user):       #gets recommended movies for a particular user
	similar_movie_list=build_similaritems(ratings)
	userMovieRatings=ratings[user]
	scores={}
	totalSimilarity={}
	for (movie,rating) in userMovieRatings.items():
		
		for (similarity,movie2) in similar_movie_list[movie]:
			
			if (movie2 in userMovieRatings):continue

			scores.setdefault(movie2,0)
			scores[movie2]+=rating*similarity

			totalSimilarity.setdefault(movie2,0)
			totalSimilarity[movie2]+=similarity

	predicted_ratings=[(score/totalSimilarity[movie],movie) for movie,score in scores.items() ]
	predicted_ratings.sort()
	predicted_ratings.reverse()
	return predicted_ratings


ratings=loadMovieLens()
#print(transformRating(ratings))
print (getMovieRecommendations(ratings,'50'))













