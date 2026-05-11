import { useMutation, useQueryClient } from "@tanstack/react-query";
import { todoApi } from "./todoApi";

export const useToggleFavorite = () => {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: ({
      id,
      favorite,
    }: {
      id: string;
      favorite: boolean;
    }) => todoApi.toggleFavorite(id, favorite),

    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["todos"] });
    },
  });
};