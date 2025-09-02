// @ts-nocheck
import { fetchPopularMovies, fetchTopRatedMovies, fetchTrendingMovies, getGenres, getMoviesByGenre } from "$lib/api/API";
import { popularMovies } from "$lib/store/GlobalState";
import type { PageServerLoad } from "./$types";

export const load = async ({ fetch }: Parameters<PageServerLoad>[0]) => {


   try {
      const [popularMovies, topRatedMovies, trendingMovies, genresResponse] = await Promise.all(
         [fetchPopularMovies(fetch),
         fetchTopRatedMovies(fetch),
         fetchTrendingMovies(fetch),
         getGenres(fetch)])


      console.log(genresResponse);


      const moviesByGenres: MoviesWithGenre[] = genresResponse
         ? await Promise.all(genresResponse.map(async (genre: Genre) => {
            const movies = await getMoviesByGenre(fetch, genre.id)
            return { id: genre.id, name: genre.name, movies }
         })) 
         : []



         
         const ramdomIndex = Math.floor(Math.random() * popularMovies.length )

         const selectedMovie = popularMovies[ramdomIndex]


         // fetch the trailer of the selected movie



         return{
            popularMovies,
            topRatedMovies,
            trendingMovies,
            moviesByGenres,
            selectedMovie
         }
         

   } catch (error) {

      console.log(error);

      return {
         popularMovies : [],
         topRatedMovies : [],
         trendingMovies : [],
         moviesByGenres : [],
         selectedMovie : null
      }


   }

}