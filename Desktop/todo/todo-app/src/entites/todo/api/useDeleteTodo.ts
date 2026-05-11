import { useMutation, useQueryClient } from "@tanstack/react-query";
import { todoApi } from "./todoApi";

export const useDeleteTodo = () => {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: todoApi.deleteTodo,

    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["todos"] });
    },
  });
};